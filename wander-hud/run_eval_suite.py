"""WanderAI demo EVAL PIPELINE — run several agents over the SAME held-out HUD
tasksets and produce ONE unified leaderboard + markdown report spanning BOTH
perception tracks:

    symbolic (2D)  — privileged geometric observation  (eval_tasks.py)
    vision   (3D)  — rendered MuJoCo RGB+depth          (eval_tasks_3d.py)

The story it tells, measured per-track in one standardized HUD env:
    random (floor)  <  our RFT model  <  Claude (ceiling)  ≤  oracle (upper bound)

Agents:
  random  — RandomBridgeAgent: uniformly random moves (the floor).
  ours    — WanderBridgeAgent driving our deployed RFT model (the contribution).
  ceiling — optional native HUD agent (e.g. claude) via --ceiling MODEL.

Usage:
  HUD_API_KEY=... FIREWORKS_API_KEY=... ../.venv-hud/bin/python run_eval_suite.py
  ../.venv-hud/bin/python run_eval_suite.py --ceiling claude          # add a ceiling run
  ../.venv-hud/bin/python run_eval_suite.py --tracks symbolic         # one track only
  ../.venv-hud/bin/python run_eval_suite.py --tracks vision --no-oracle
"""
import argparse
import asyncio
import math
import os
import statistics

from hud import Taskset
from hud.agents.types import OpenAIChatConfig

from bridge_agent import WanderBridgeAgent
from eval_agents import RandomBridgeAgent
from eval_tasks import EVAL_SEEDS
from eval_tasks_3d import VISION_SCENES

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FW_BASE = "https://api.fireworks.ai/inference/v1"
SUCCESS = 0.99          # reward >= this counts as "reached the ball"

# Per-track agent step budgets. HUD's loop defaults to max_steps=10, but our rooms
# need many more to reach the ball (the env allows far more). The 3D rooms are
# larger, so they need a bigger budget than the compact 2D rooms.
TRACKS = {
    "symbolic": {"label": "symbolic (2D)", "module": "eval_tasks.py", "max_steps": 60},
    "vision":   {"label": "vision (3D)",   "module": "eval_tasks_3d.py", "max_steps": 110},
}


def _trained_model() -> str:
    m = os.environ.get("WANDER_TRAINED_MODEL")
    if m:
        return m
    p = os.path.join(_REPO, ".trained_model.txt")
    if os.path.exists(p):
        return open(p).read().strip()
    raise SystemExit("set WANDER_TRAINED_MODEL or create .trained_model.txt")


def _fw_config(model: str, max_steps: int) -> OpenAIChatConfig:
    cfg = OpenAIChatConfig(model=model, api_key=os.environ["FIREWORKS_API_KEY"], base_url=_FW_BASE)
    try:                       # let episodes run long enough to actually reach the ball
        cfg.max_steps = max_steps
    except Exception:
        pass
    return cfg


def _run_rewards(job):
    """Per-task rewards from a finished Job (best-effort across hud versions)."""
    rewards = []
    for r in getattr(job, "runs", []) or []:
        v = getattr(r, "reward", None)
        rewards.append(float(v) if v is not None else 0.0)
    return rewards


def _summary(label, rewards, n, job=None):
    return {
        "label": label,
        "mean": round(statistics.mean(rewards), 3) if rewards else 0.0,
        "success": sum(1 for r in rewards if r >= SUCCESS),
        "n": len(rewards) or n,
        "rewards": [round(r, 2) for r in rewards],
        "job_id": getattr(job, "id", None) if job else None,
    }


# ---- oracle upper bound (local, privileged geodesic policy) ----
def _oracle_rewards(envs, max_steps):
    """Run the privileged OraclePolicy in each (already-reset-capable) env and score
    it like the HUD template (success=1 else fraction of geodesic closed)."""
    from wanderai.policies import OraclePolicy
    rewards = []
    for e in envs:
        _, info = e.reset()
        opt, fg, success = info["optimal"], info["geodesic"], False
        pol = OraclePolicy()
        for _ in range(max_steps):
            _, _, done, info = e.step(int(pol.act(None, e)))
            fg, success = info["geodesic"], info["success"]
            if done:
                break
        reward = 1.0 if success else (max(0.0, min(1.0, (opt - fg) / opt))
                                      if opt > 0 and math.isfinite(fg) else 0.0)
        rewards.append(reward)
    return rewards


def _oracle_2d(max_steps):
    from wanderai.scene_gen import make_split
    from wanderai.environment import SceneSearchEnv, EnvConfig
    envs = [SceneSearchEnv(make_split(0, 1, seed=s)[1][0], config=EnvConfig(max_steps=max_steps))
            for s in EVAL_SEEDS]
    return _oracle_rewards(envs, max_steps)


def _oracle_3d(max_steps):
    from wanderai.environment import SceneSearchEnv, EnvConfig
    from env import _load_3d                       # reuse the cached 3D scenes the HUD env serves
    envs = []
    for s in VISION_SCENES:
        scene, renderer = _load_3d(s)
        envs.append(SceneSearchEnv(scene, renderer=renderer,
                                   config=EnvConfig(max_steps=max_steps, perception="vision")))
    return _oracle_rewards(envs, max_steps)


# ---- deployed policy reference: the trained model + geodesic safety net ----
def _assisted_rewards(envs, max_steps, model):
    """Run the DEPLOYED policy (AssistedLLMPolicy = our RFT model + a geodesic
    safety-net that breaks ping-pong/stuck loops the myopic single-step model
    falls into) and score like the HUD template. Uses the privileged distance
    field for the nudge, so — like the oracle — it's a local reference, not a pure
    HUD agent; it's what the UI 'Trained' button actually runs."""
    from wanderai.llm_policy import AssistedLLMPolicy
    rewards = []
    for e in envs:
        _, info = e.reset()
        opt, fg, success = info["optimal"], info["geodesic"], False
        pol = AssistedLLMPolicy(model=model)
        for _ in range(max_steps):
            _, _, done, info = e.step(int(pol.act(None, e)))
            fg, success = info["geodesic"], info["success"]
            if done:
                break
        reward = 1.0 if success else (max(0.0, min(1.0, (opt - fg) / opt))
                                      if opt > 0 and math.isfinite(fg) else 0.0)
        rewards.append(reward)
    return rewards


def _assisted_2d(max_steps, model):
    from wanderai.scene_gen import make_split
    from wanderai.environment import SceneSearchEnv, EnvConfig
    envs = [SceneSearchEnv(make_split(0, 1, seed=s)[1][0], config=EnvConfig(max_steps=max_steps))
            for s in EVAL_SEEDS]
    return _assisted_rewards(envs, max_steps, model)


def _assisted_3d(max_steps, model):
    from wanderai.environment import SceneSearchEnv, EnvConfig
    from env import _load_3d
    envs = []
    for s in VISION_SCENES:
        scene, renderer = _load_3d(s)
        envs.append(SceneSearchEnv(scene, renderer=renderer,
                                   config=EnvConfig(max_steps=max_steps, perception="vision")))
    return _assisted_rewards(envs, max_steps, model)


async def _eval_agent_on_track(label, agent, module, max_concurrent):
    ts = Taskset.from_module(module)
    n = len(list(ts.items()))
    print(f"   · {label}: {n} rooms …")
    job = await ts.run(agent, group=1, max_concurrent=max_concurrent)
    return _summary(label, _run_rewards(job), n, job)


def _build_agents(args, max_steps):
    """Yield (label, agent_factory) — factory(max_steps) so each track gets its own
    step budget (Claude/native agents are built once)."""
    model = _trained_model()
    short = model.split("/models/")[-1].split("#")[0]
    out = [
        ("random (floor)", lambda ms: RandomBridgeAgent(config=_fw_config(model, ms), seed=0)),
        (f"ours · {short}", lambda ms: WanderBridgeAgent(config=_fw_config(model, ms), wander_model=model)),
    ]
    if args.ceiling:
        def _ceiling(ms, name=args.ceiling):
            from hud.agents import create_agent
            return create_agent(name)
        out.append((f"ceiling · {args.ceiling}", _ceiling))
    return model, out


def _write_report(track_keys, results, refs, model):
    """results[track][label] = HUD-agent summary; refs[track][label] = privileged
    local-reference summary (deployed policy + oracle)."""
    def cell(s):
        return f"{s['mean']:.3f} ({s['success']}/{s['n']})" if s else "—"

    def overall(lab, src):
        rs = [src[tk][lab]["mean"] for tk in track_keys if src.get(tk, {}).get(lab)]
        return round(statistics.mean(rs), 3) if rs else None

    def rows_for(src, italic):
        labels = dict.fromkeys(lab for tk in track_keys for lab in src.get(tk, {}))
        out = []
        for lab in labels:
            cells = [cell(src.get(tk, {}).get(lab)) for tk in track_keys]
            ov = overall(lab, src)
            name = f"_{lab}_" if italic else f"**{lab}**" if lab.startswith("ours ·") else lab
            ovs = (f"_{ov:.3f}_" if italic else f"**{ov:.3f}**") if ov is not None else "—"
            out.append((ov if ov is not None else -1,
                        f"| {name} | " + " | ".join(cells) + f" | {ovs} |"))
        return [r for _, r in sorted(out)]

    head = "| Agent | " + " | ".join(TRACKS[tk]["label"] for tk in track_keys) + " | Overall |"
    sep = "|" + "---|" * (len(track_keys) + 2)
    lines = [
        "# WanderAI — Unified HUD Evaluation Leaderboard",
        "",
        "One standardized HUD env, two perception tracks: **symbolic (2D)** = privileged "
        "geometric observation; **vision (3D)** = the agent perceives a real MuJoCo room "
        "through rendered RGB+depth. Reward = 1.0 if the agent reaches the red ball, else the "
        "fraction of geodesic distance it closed (0–1).",
        "",
        f"- Symbolic track: **{len(EVAL_SEEDS)} held-out rooms** (seeds {EVAL_SEEDS[0]}–{EVAL_SEEDS[-1]}, none trained on).",
        f"- Vision track: **{len(VISION_SCENES)} MuJoCo rooms** ({', '.join(VISION_SCENES)}); 'test' is held out from training.",
        "",
        "**HUD agents** (drive the env purely from the observation):",
        "",
        head, sep,
    ]
    lines += rows_for(results, italic=False)
    lines += [
        "",
        "**Local references** (privileged — use the ground-truth distance field, so not pure "
        "HUD agents): _ours+assist_ is the deployed UI policy = the RFT model + a geodesic "
        "safety-net that breaks the ping-pong loops the myopic single-step model falls into; "
        "_oracle_ is the geodesic-optimal upper bound.",
        "",
        head, sep,
    ]
    lines += rows_for(refs, italic=True)

    lines += ["", "## HUD trace jobs", ""]
    for tk in track_keys:
        for lab, s in results.get(tk, {}).items():
            if s.get("job_id"):
                lines.append(f"- **{lab}** · {TRACKS[tk]['label']} — HUD job `{s['job_id']}`")
    lines += ["", f"_Our model: `{model}`. Floor = uniformly random moves. "
              "Generated by `wander-hud/run_eval_suite.py`._"]

    path = os.path.join(_REPO, "docs", "eval_report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ceiling", default=os.environ.get("WANDER_CEILING"),
                    help="optional native HUD agent for the ceiling (e.g. 'claude')")
    ap.add_argument("--tracks", default="symbolic,vision",
                    help="comma list of tracks to run: symbolic, vision")
    ap.add_argument("--no-oracle", action="store_true", help="skip the local oracle upper bound")
    ap.add_argument("--no-assist", action="store_true",
                    help="skip the deployed policy (RFT + geodesic safety net) reference")
    ap.add_argument("--max-concurrent", type=int, default=3)
    args = ap.parse_args()

    track_keys = [t.strip() for t in args.tracks.split(",") if t.strip() in TRACKS]
    if not track_keys:
        raise SystemExit(f"no valid tracks in '{args.tracks}'; choose from {list(TRACKS)}")

    model = _trained_model()
    short = model.split("/models/")[-1].split("#")[0]
    results = {tk: {} for tk in track_keys}
    refs = {tk: {} for tk in track_keys}
    for tk in track_keys:
        spec = TRACKS[tk]
        print(f"\n=== TRACK: {spec['label']} (max_steps={spec['max_steps']}) ===")
        _, agents = _build_agents(args, spec["max_steps"])
        for label, factory in agents:
            try:
                results[tk][label] = await _eval_agent_on_track(
                    label, factory(spec["max_steps"]), spec["module"], args.max_concurrent)
            except Exception as e:
                print(f"   ✗ {label} failed: {type(e).__name__}: {e}")
        # local references (privileged): the deployed policy (model + safety net)
        # and the oracle upper bound.
        if not args.no_assist:
            try:
                print(f"   · ours+assist · {short} (deployed · local) …")
                rw = (_assisted_2d if tk == "symbolic" else _assisted_3d)(spec["max_steps"], model)
                refs[tk][f"ours+assist · {short} (deployed · local)"] = _summary("assist", rw, len(rw))
            except Exception as e:
                print(f"   ✗ assist failed: {type(e).__name__}: {e}")
        if not args.no_oracle:
            try:
                print("   · oracle (upper bound · local) …")
                rw = (_oracle_2d if tk == "symbolic" else _oracle_3d)(spec["max_steps"])
                refs[tk]["oracle (upper bound · local)"] = _summary("oracle", rw, len(rw))
            except Exception as e:
                print(f"   ✗ oracle failed: {type(e).__name__}: {e}")

    print("\n=== UNIFIED LEADERBOARD ===")
    print(f"{'agent':40s}" + "".join(f"  {TRACKS[tk]['label']:>16s}" for tk in track_keys))
    for src in (results, refs):
        labels = dict.fromkeys(lab for tk in track_keys for lab in src.get(tk, {}))
        for lab in labels:
            cells = []
            for tk in track_keys:
                s = src[tk].get(lab)
                cells.append(f"{s['mean']:.3f} {s['success']}/{s['n']}" if s else "—")
            print(f"{lab:40s}" + "".join(f"  {c:>16s}" for c in cells))

    path = _write_report(track_keys, results, refs, model)
    print(f"\n📄 unified report written: {path}")


if __name__ == "__main__":
    asyncio.run(main())
