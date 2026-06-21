from __future__ import annotations
from dataclasses import dataclass
from .geometry import AABB, Pose


@dataclass(frozen=True)
class Scene:
    bounds: AABB
    obstacles: list[AABB]
    ball: tuple[float, float]
    agent_start: Pose
    agent_radius: float = 0.2

    def is_free(self, x: float, y: float) -> bool:
        """A point is free if it is inside the room and outside every obstacle
        inflated by the agent radius (configuration-space test)."""
        if not self.bounds.contains(x, y):
            return False
        for ob in self.obstacles:
            if ob.inflate(self.agent_radius).contains(x, y):
                return False
        return True


def default_scene() -> Scene:
    """A 6x6 m room with three box obstacles; the ball sits in a corner partly
    occluded by a box, with the agent starting in the opposite corner."""
    return Scene(
        bounds=AABB(0, 0, 6, 6),
        obstacles=[
            AABB(2.0, 2.0, 3.0, 4.5),   # vertical divider
            AABB(3.8, 1.0, 4.6, 2.2),   # box near the ball (occluder)
            AABB(1.0, 4.8, 4.0, 5.2),   # upper wall stub
        ],
        ball=(5.4, 1.4),
        agent_start=Pose(0.6, 0.6, 0.0),
        agent_radius=0.2,
    )
