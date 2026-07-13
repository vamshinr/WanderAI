"""Local episodic RL: GRPO-style REINFORCE over the symbolic observation.

This mirrors the Fireworks episodic-RFT loop *locally* — pure numpy, no API key,
fully seeded — so the paper's "episodic RL improves search" claim is backed by
numbers anyone can reproduce on a laptop. The learner is deliberately tiny (a
one-hidden-layer MLP over the same egocentric features the text policy reads);
the point is not architecture, it is the training signal: group-relative
advantages (GRPO, Shao et al. 2024, "DeepSeekMath") computed from groups of
episodes rolled out in the *same* room, which cancels the huge per-room return
variance that plain REINFORCE drowns in.

Honesty contract — features come ONLY from:
  * the egocentric `Observation` (ball visibility/bearing with real occlusion,
    ray-cast clearance, visited-cell bits — the channel every WanderAI policy reads);
  * the policy's own last action;
  * optionally a `frontier_hint` from an `EgoMap` the rollout builds from its OWN
    depth strips, exactly like `FrontierPolicy` (the agent's own map summary).
Never `env.field`, `env.grid`, `scene.ball`, or `scene.obstacles`. The *reward*
is privileged (geodesic shaping) — that is standard: reward is the trainer's
signal, not the agent's sensor, and is unavailable at deployment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math

import numpy as np

from .environment import Action, EnvConfig, SceneSearchEnv
from .mapping import EgoMap, depth_strip, frontier_hint
from .metrics import summarize
from .observation import Observation, observe
from .policies import run_episode
from .scene import Scene
from .scene_gen import make_split

# Feature layout (keep in sync with featurize()):
#   [0]     ball visible (0/1)
#   [1:3]   sin/cos of ball bearing (0,0 when not visible — NOT cos(0)=1)
#   [3]     min(ball distance, 8)/8, 0 when not visible
#   [4:7]   clearance left/center/right, /6 clipped to [0,1]
#   [7:10]  explored left/center/right (0/1)
#   [10:13] last-action one-hot (all 0 on the first step)
#   [13:17] frontier-hint direction one-hot: left/center/right/behind (all 0 if None)
#   [17]    min(hint distance, 8)/8, 0 if no hint
#   [18]    bias 1.0
FEATURE_DIM = 19
_HINT_DIRS = ("left", "center", "right", "behind")
_MAX_DIST = 8.0
_MAX_CLEAR = 6.0
N_ACTIONS = 3


def featurize(obs: Observation, last_action: int | None,
              hint: dict | None) -> np.ndarray:
    """Egocentric observation -> fixed-size float32 vector. Every entry is in
    [-1, 1]; absent signals are all-zeros so the net can gate on the indicator."""
    x = np.zeros(FEATURE_DIM, dtype=np.float32)
    if obs.ball_visible and obs.ball_bearing is not None:
        x[0] = 1.0
        x[1] = math.sin(obs.ball_bearing)
        x[2] = math.cos(obs.ball_bearing)
        x[3] = min(obs.ball_distance, _MAX_DIST) / _MAX_DIST
    for k, side in enumerate(("left", "center", "right")):
        x[4 + k] = min(obs.clearance[side], _MAX_CLEAR) / _MAX_CLEAR
        x[7 + k] = 1.0 if obs.explored[side] else 0.0
    if last_action is not None:
        x[10 + int(last_action)] = 1.0
    if hint is not None:
        x[13 + _HINT_DIRS.index(hint["direction"])] = 1.0
        x[17] = min(hint["distance"], _MAX_DIST) / _MAX_DIST
    x[18] = 1.0
    return x


class PolicyNet:
    """Tiny numpy MLP: FEATURE_DIM -> hidden (tanh) -> 3 action logits.

    Hidden layer is He-initialized (seeded); the OUTPUT layer starts at zero so
    the initial policy is exactly uniform — the standard policy-gradient choice:
    maximal exploration at step 0 and no arbitrary initial action bias that
    training would first have to unlearn."""

    def __init__(self, hidden: int = 32, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.hidden = hidden
        # He init scaled for tanh (fan-in); float64 keeps Adam bit-stable.
        self.W1 = rng.normal(0.0, math.sqrt(2.0 / FEATURE_DIM),
                             (hidden, FEATURE_DIM))
        self.b1 = np.zeros(hidden)
        self.W2 = np.zeros((N_ACTIONS, hidden))
        self.b2 = np.zeros(N_ACTIONS)

    # --- forward ---
    def _forward(self, X: np.ndarray):
        """X (N, DIM) -> (hidden activations (N, H), logits (N, 3))."""
        H = np.tanh(X @ self.W1.T + self.b1)
        return H, H @ self.W2.T + self.b2

    def logits(self, x: np.ndarray) -> np.ndarray:
        _, z = self._forward(np.asarray(x, dtype=np.float64)[None, :])
        return z[0]

    def sample(self, x: np.ndarray, rng: np.random.Generator):
        """Sample an action; returns (action, logp). Numerically stable softmax."""
        z = self.logits(x)
        z = z - z.max()
        p = np.exp(z)
        p /= p.sum()
        a = int(rng.choice(N_ACTIONS, p=p))
        return a, float(math.log(max(p[a], 1e-12)))

    def argmax(self, x: np.ndarray) -> int:
        return int(np.argmax(self.logits(x)))

    # --- (de)serialization ---
    def to_dict(self) -> dict:
        return {"hidden": self.hidden,
                "W1": self.W1.tolist(), "b1": self.b1.tolist(),
                "W2": self.W2.tolist(), "b2": self.b2.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "PolicyNet":
        net = cls(hidden=int(d["hidden"]))
        net.W1 = np.asarray(d["W1"], dtype=np.float64)
        net.b1 = np.asarray(d["b1"], dtype=np.float64)
        net.W2 = np.asarray(d["W2"], dtype=np.float64)
        net.b2 = np.asarray(d["b2"], dtype=np.float64)
        return net

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


class _Adam:
    """Adam (Kingma & Ba 2015) over a list of numpy parameter arrays, in place."""

    def __init__(self, params: list[np.ndarray], lr: float,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list[np.ndarray], grads: list[np.ndarray]):
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            mhat = m / (1 - self.b1 ** self.t)
            vhat = v / (1 - self.b2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


@dataclass
class RLConfig:
    iters: int = 400
    group_size: int = 8          # G episodes per GRPO group (same room)
    lr: float = 3e-3
    entropy_coef: float = 0.01
    max_steps: int = 120         # per training episode
    hidden: int = 32
    seed: int = 0
    use_hint: bool = False       # append the agent's own frontier hint to features
    adv_clip: float = 4.0


@dataclass
class _EpisodeTrace:
    features: list = field(default_factory=list)   # per-step float32 vectors
    actions: list = field(default_factory=list)    # per-step ints
    logps: list = field(default_factory=list)      # sampled logp (diagnostics)
    ret: float = 0.0                               # undiscounted sum of rewards
    success: bool = False


class ReinforceTrainer:
    """Group-relative REINFORCE (the GRPO advantage of Shao et al. 2024, minus
    the ratio clipping — one on-policy gradient step per group):

        A_i = (R_i - mean_j R_j) / (std_j R_j + 1e-6), clipped to +-adv_clip

    Each iteration rolls out `group_size` stochastic episodes in the SAME room
    (round-robin over training scenes; the env is cached per scene since reset()
    memoizes the occupancy grid and distance field), then takes ONE Adam step on
    the mean over steps of  A_i * grad log pi(a_t|x_t) + entropy_coef * grad H."""

    def __init__(self, scenes: list[Scene], cfg: RLConfig | None = None):
        self.cfg = cfg or RLConfig()
        self.net = PolicyNet(hidden=self.cfg.hidden, seed=self.cfg.seed)
        self.rng = np.random.default_rng(self.cfg.seed)
        self.opt = _Adam(self.net.params(), lr=self.cfg.lr)
        self._envs = [SceneSearchEnv(s, config=EnvConfig(max_steps=self.cfg.max_steps))
                      for s in scenes]
        self.history: list[dict] = []

    # --- rollout (only honest channels; mirrors TrainedLocalPolicy.act) ---
    def _rollout(self, env: SceneSearchEnv) -> _EpisodeTrace:
        env.reset()
        trace = _EpisodeTrace()
        last_action: int | None = None
        egomap = EgoMap() if self.cfg.use_hint else None
        done = False
        info = {"success": False}
        while not done:
            obs = observe(env.scene, env.pose, history=env.history,
                          visited=env.visited)
            hint = None
            if egomap is not None:
                egomap.update(env.pose, depth_strip(env.scene, env.pose))
                hint = frontier_hint(egomap, env.pose)
            x = featurize(obs, last_action, hint)
            a, logp = self.net.sample(x, self.rng)
            trace.features.append(x)
            trace.actions.append(a)
            trace.logps.append(logp)
            _, reward, done, info = env.step(a)
            trace.ret += reward
            last_action = a
        trace.success = bool(info["success"])
        return trace

    # --- one GRPO update from a group of episodes ---
    def update_from_group(self, episodes: list[_EpisodeTrace]) -> dict:
        returns = np.array([e.ret for e in episodes])
        adv = (returns - returns.mean()) / (returns.std() + 1e-6)
        adv = np.clip(adv, -self.cfg.adv_clip, self.cfg.adv_clip)

        X = np.concatenate([np.asarray(e.features, dtype=np.float64)
                            for e in episodes])                        # (N, DIM)
        acts = np.concatenate([e.actions for e in episodes]).astype(int)
        A = np.concatenate([np.full(len(e.actions), a)
                            for e, a in zip(episodes, adv)])           # (N,)
        n = len(acts)

        H, Z = self.net._forward(X)
        Z = Z - Z.max(axis=1, keepdims=True)
        P = np.exp(Z)
        P /= P.sum(axis=1, keepdims=True)
        logP = np.log(np.maximum(P, 1e-12))
        ent = -(P * logP).sum(axis=1)                                  # (N,)

        # Ascend J = mean[ A * log pi(a|x) + c * H ]. dJ/dlogits per step:
        one_hot = np.zeros_like(P)
        one_hot[np.arange(n), acts] = 1.0
        dZ = (A[:, None] * (one_hot - P)
              - self.cfg.entropy_coef * P * (logP + ent[:, None])) / n

        # Backprop through logits = H W2^T + b2, H = tanh(X W1^T + b1).
        gW2 = dZ.T @ H
        gb2 = dZ.sum(axis=0)
        dH = (dZ @ self.net.W2) * (1.0 - H * H)
        gW1 = dH.T @ X
        gb1 = dH.sum(axis=0)

        grads = [gW1, gb1, gW2, gb2]
        grad_norm = math.sqrt(sum(float((g * g).sum()) for g in grads))
        # Adam minimizes, we ascend J: feed the negated gradient.
        self.opt.step(self.net.params(), [-g for g in grads])

        return {"mean_return": float(returns.mean()),
                "success_rate": float(np.mean([e.success for e in episodes])),
                "mean_entropy": float(ent.mean()),
                "grad_norm": grad_norm}

    def train(self, progress=None) -> list[dict]:
        """Run cfg.iters GRPO iterations; returns (and stores) per-iter history.
        `progress(iter_index, stats)` is called after every iteration if given."""
        for it in range(self.cfg.iters):
            env = self._envs[it % len(self._envs)]
            episodes = [self._rollout(env) for _ in range(self.cfg.group_size)]
            stats = self.update_from_group(episodes)
            stats["iter"] = it
            self.history.append(stats)
            if progress is not None:
                progress(it, stats)
        return self.history


class TrainedLocalPolicy:
    """Deployment policy: `act(obs, env) -> Action` via argmax over the trained
    net, wrapped in the same two unstuck reflexes `FrontierPolicy` uses —
    necessary because a deterministic argmax of a reactive net can enter limit
    cycles (spin forever, ram a wall) that the stochastic training policy never
    exhibits:

      * collision inference — a MOVE_FORWARD that did not change the pose marks
        the intended destination cell blocked; forward is masked while it would
        re-enter a blocked cell (or the depth clearance shows no room);
      * anti-dither — after a full circle of consecutive turns, force one step
        forward when it is safe.

    Both reflexes read only the agent's own pose history and its observation, so
    the honesty contract holds. `train_and_eval` applies the IDENTICAL wrapper to
    the untrained and the trained weights, so the reported improvement is
    attributable to learning, not to the reflexes.

    Self-resets its episode state (last action, blocked cells, own EgoMap) on
    env.steps==0, like every other WanderAI policy."""

    BLOCK_CELL = 0.25    # resolution of the collision-inferred blocked-cell set

    def __init__(self, net: PolicyNet | dict, use_hint: bool = False):
        self.net = net if isinstance(net, PolicyNet) else PolicyNet.from_dict(net)
        self.use_hint = use_hint
        self._reset()

    @classmethod
    def from_file(cls, path: str, use_hint: bool = False) -> "TrainedLocalPolicy":
        with open(path) as f:
            return cls(json.load(f), use_hint=use_hint)

    def _reset(self):
        self.last_action: int | None = None
        self.last_pose = None
        self.blocked: set = set()
        self.consecutive_turns = 0
        self.map = EgoMap() if self.use_hint else None

    def _ahead_cell(self, pose, step_size: float) -> tuple[int, int]:
        ax = pose.x + step_size * math.cos(pose.heading)
        ay = pose.y + step_size * math.sin(pose.heading)
        return (int(math.floor(ax / self.BLOCK_CELL)),
                int(math.floor(ay / self.BLOCK_CELL)))

    def act(self, obs, env: SceneSearchEnv) -> Action:
        if env.steps == 0:
            self._reset()
        pose, ecfg = env.pose, env.config

        # Collision inference: we asked to move and the pose did not change.
        if (self.last_action == int(Action.MOVE_FORWARD)
                and self.last_pose is not None
                and math.hypot(pose.x - self.last_pose.x,
                               pose.y - self.last_pose.y) < 1e-9):
            self.blocked.add(self._ahead_cell(self.last_pose, ecfg.step_size))

        sym = observe(env.scene, pose, history=env.history, visited=env.visited)
        hint = None
        if self.map is not None:
            self.map.update(pose, depth_strip(env.scene, pose))
            hint = frontier_hint(self.map, pose)

        forward_ok = (self._ahead_cell(pose, ecfg.step_size) not in self.blocked
                      and sym.clearance["center"] > ecfg.step_size * 1.2)
        z = self.net.logits(featurize(sym, self.last_action, hint))
        if not forward_ok:
            z = z.copy()
            z[int(Action.MOVE_FORWARD)] = -1e9
        a = int(np.argmax(z))
        if a != int(Action.MOVE_FORWARD):
            self.consecutive_turns += 1
            if (self.consecutive_turns > int(2 * math.pi / ecfg.turn)
                    and forward_ok):
                a = int(Action.MOVE_FORWARD)
        if a == int(Action.MOVE_FORWARD):
            self.consecutive_turns = 0

        self.last_pose, self.last_action = pose, a
        return Action(a)


def train_and_eval(n_train: int = 12, n_test: int = 8, split_seed: int = 7,
                   cfg: RLConfig | None = None, progress=None) -> dict:
    """The one call the benchmark uses: train on `n_train` rooms, evaluate the
    SAME initial weights (untrained) and the trained weights on `n_test`
    held-out rooms with EnvConfig(max_steps=300). Fully deterministic given
    (n_train, n_test, split_seed, cfg)."""
    cfg = cfg or RLConfig()
    train_scenes, test_scenes = make_split(n_train, n_test, seed=split_seed)
    trainer = ReinforceTrainer(train_scenes, cfg)
    untrained_weights = trainer.net.to_dict()   # snapshot before any update

    trainer.train(progress=progress)

    def _eval(weights: dict) -> dict:
        policy = TrainedLocalPolicy(weights, use_hint=cfg.use_hint)
        results = [run_episode(SceneSearchEnv(s, config=EnvConfig(max_steps=300)),
                               policy) for s in test_scenes]
        return summarize(results)

    return {"train_history": trainer.history,
            "eval_untrained": _eval(untrained_weights),
            "eval_trained": _eval(trainer.net.to_dict()),
            "weights": trainer.net.to_dict()}
