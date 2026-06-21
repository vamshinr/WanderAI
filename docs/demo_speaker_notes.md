# WanderAI — 3-minute demo speaker notes

**Total: ~3:00.** Read the **SAY** lines at a natural pace (~150 wpm). **SHOW** = what to
do on screen. Recommended screen: the UI (local `python3 serve.py` → localhost:8000, or
the hosted link https://vamshinr-wanderai.hf.space) on a **2D Default scene**.

### Before you hit record (setup checklist)
- UI open, **Default scene** loaded (so the room + heatmap are already on screen).
- Confirm the **st1** Fireworks deployment is live (the Trained button works).
- Have the **Model dropdown** on **Auto** (uses st1 for 2D).
- Do one dry run of base→trained so you know the timing.

---

### [0:00–0:35] Problem statement
**SHOW:** The 2D room on screen — agent (blue), red ball, obstacles, the FOV cone.

**SAY:** "This is WanderAI. The problem: drop an agent into a room it has *never seen*,
give it only a first-person view, and have it find a red ball — then have that search
behavior generalize to brand-new rooms. The hard part is partial observability: the
agent can't see through walls, so it can't just beeline to the goal. It has to actually
*explore*, and learn a search strategy that transfers across scenes. We built it for the
HUD hackathon, training on Fireworks, with 3D rooms from Antim Labs."

### [0:35–1:15] The rules — observation, actions, assumptions
**SHOW:** Point at the side panel (the symbolic observation text) and the clearance numbers.

**SAY:** "Here are the rules. The agent never gets the map. Each step it sees a compact
observation: whether the red ball is in its line of sight — and *only then* its bearing
and distance — plus the open space to its left, center, and right, and which directions
it has already explored. It picks one of three actions: forward, turn left, turn right.
The key constraint: it only learns where the ball is once the ball comes into view — so
it's *forced* to search. And at run time it uses nothing privileged — no map, no
shortest-path oracle, no hidden ball location. Only what it can perceive."

### [1:15–1:45] The reward (high level)
**SHOW:** The geodesic heatmap on the map (bright = close to the ball).

**SAY:** "For training, the reward is based on *geodesic* distance — the shortest
*walkable* path that routes around obstacles, not straight-line distance. That's
important: straight-line distance would reward walking into walls; the geodesic reward
always points along a route the agent can actually take. Each step it's rewarded for
getting closer, lightly penalized for time and for collisions, with a bonus for reaching
the ball. And this reward is *privileged* — it's only used during training. The agent
itself never sees it."

### [1:45–2:35] Demo — base vs RFT-trained, same room
**SHOW:** Model dropdown → pick base. Click **▶ LLM (base)**. Let it run ~10–15 steps.

**SAY:** "Now the demo — same room, two models. First the *untrained* base model. Watch —
it drifts, it doesn't really search, and it doesn't find the ball."

**SHOW:** Click **■ Stop**. Switch to **Trained** (st1). Click **▶ Trained**.

**SAY:** "Now our reinforcement-fine-tuned model, same room. See the amber line — that's
its actual path — and the green cells are where it's already been. Notice it spreads into
*new* ground instead of looping, turns to scan, and the moment the ball comes into view it
homes in and reaches it. Same observation, same three actions — the *only* difference is
reinforcement fine-tuning on Fireworks. That's RFT teaching the model to explore."

> Fallback if the base accidentally stumbles toward the ball: "Even when it wanders the
> right way, it's aimless — the trained model searches *deliberately* and covers new ground."

### [2:35–3:00] Future plans + close
**SHOW:** (Optional) flash the 3D scene or the HUD leaderboard, then back to the room.

**SAY:** "What's next. We've prototyped *multi-turn* RFT — training on whole episodes, so
the model optimizes the entire search rather than one step at a time. There's a 3D vision
version with real MuJoCo rooms, and a HUD leaderboard that scores us between a random
floor and an optimal ceiling. The bet: train on a handful of rooms and generalize
everywhere. Thanks for watching."

---

**Timing tip:** if you're running long, trim the reward section to one sentence
("rewarded for getting closer, with penalties for time and collisions — and it's
privileged, used only in training"). The demo (1:45–2:35) is the part to protect.
