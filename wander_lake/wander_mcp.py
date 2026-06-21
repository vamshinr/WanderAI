"""WanderAI scene-search as an eval-protocol McpGym env (multi-turn RFT).

The model drives the agent by calling the `move` tool each turn; the base McpGym
tracks reward/termination in the control plane so the trainer (GRPO) sees the
whole-episode return."""
from typing import Any, Dict, Optional

from mcp.server.fastmcp import Context
from eval_protocol.mcp import McpGym

from wander_adapter import WanderAdapter


class WanderMcp(McpGym):
    def __init__(self, seed: Optional[int] = None, **kwargs):
        super().__init__("wander-scene-search", WanderAdapter(), seed, **kwargs)

    def _register_tools(self):
        @self.mcp.tool(
            name="move",
            description=("Move the agent one step. action must be MOVE_FORWARD, TURN_LEFT, "
                        "or TURN_RIGHT. Returns the new first-person observation (whether the "
                        "red ball is visible + its bearing/distance, and clearance left/center/"
                        "right). Keep calling until you reach the red ball."),
        )
        def move(action: str, ctx: Context) -> Dict[str, Any]:
            action = (action or "").strip().upper()
            action_int = self.adapter.parse_action(action)
            session_id = self._get_session_id(ctx)
            self._get_or_create_session(ctx)
            observation_data = self._execute_session_environment_step(session_id, action_int)
            observation_data["action"] = action
            return observation_data

    def format_observation(self, obs: Any, env: Any) -> Dict[str, Any]:
        return {"observation": obs}
