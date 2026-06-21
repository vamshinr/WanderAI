"""Vision perception: red-ball segmentation + depth-based clearance, on synthetic
RGB/depth frames (no MuJoCo needed, so these run anywhere)."""
import math
import numpy as np

from wanderai.perception import (red_mask, largest_blob, detect_ball, perceive,
                                 MIN_BLOB_PIXELS)
from wanderai.geometry import Pose


def _gray(h=144, w=192, level=110):
    return np.full((h, w, 3), level, dtype=np.uint8)


def _disk(img, cy, cx, r, color=(220, 25, 25)):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    img[m] = color
    return m


def test_red_mask_picks_red_not_gray():
    img = _gray()
    _disk(img, 70, 96, 8)
    m = red_mask(img)
    assert m[70, 96]
    assert not m[10, 10]                 # gray background isn't red


def test_largest_blob_returns_biggest_component():
    mask = np.zeros((50, 50), dtype=bool)
    mask[5:7, 5:7] = True                # 4 px
    mask[20:30, 20:30] = True            # 100 px
    ys, xs = largest_blob(mask)
    assert len(xs) == 100


def test_detect_ball_accepts_disk_rejects_speck_and_bar():
    img = _gray()
    _disk(img, 72, 96, 9)                # a round ball
    hit = detect_ball(img)
    assert hit is not None
    col, ys, xs = hit
    assert abs(col - 96) < 3            # centroid ~ centre

    speck = _gray(); _disk(speck, 72, 96, 1)              # ~ a couple px
    assert detect_ball(speck) is None

    bar = _gray(); bar[70:73, 10:120] = (220, 25, 25)     # long thin red bar (furniture)
    assert detect_ball(bar) is None


def test_detect_ball_bearing_sign_left_is_positive():
    # blob on the LEFT half of the image -> positive bearing (left), matching the
    # geometric observation's convention.
    img = _gray(); _disk(img, 72, 40, 9)
    col, ys, xs = detect_ball(img)
    w = img.shape[1]
    ndc = 2 * col / w - 1
    assert ndc < 0                       # left of centre


class _FakeRenderer:
    """Minimal stand-in: returns a fixed RGB+depth so perceive() is testable
    without MuJoCo."""
    max_depth = 10.0
    fov_x = math.pi / 2

    def __init__(self, rgb, depth):
        self._rgb, self._depth = rgb, depth

    def render_rgb_depth(self, scene, pose):
        return self._rgb, self._depth


def test_perceive_reads_ball_and_clearance_from_pixels():
    rgb = _gray(); _disk(rgb, 72, 96, 10)
    depth = np.full((144, 192), 9.0, dtype=np.float32)
    # ball pixels close (2 m); a near wall on the right third (1.2 m)
    depth[red_mask(rgb)] = 2.0
    depth[:, 128:] = 1.2
    obs = perceive(_FakeRenderer(rgb, depth), scene=None,
                   pose=Pose(0.0, 0.0, 0.0), history=[], visited=set())
    assert obs.ball_visible
    assert abs(obs.ball_distance - 2.0) < 0.5
    assert obs.clearance["right"] < obs.clearance["left"]    # wall is on the right


def test_perceive_no_ball_when_no_red():
    rgb = _gray()
    depth = np.full((144, 192), 8.0, dtype=np.float32)
    obs = perceive(_FakeRenderer(rgb, depth), scene=None,
                   pose=Pose(0.0, 0.0, 0.0), history=[], visited=set())
    assert not obs.ball_visible
    assert obs.ball_bearing is None
