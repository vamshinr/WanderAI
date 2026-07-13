"""VLM policy tests — no network, no mujoco: the transport is always stubbed and
frames come from the dependency-free StubRenderer."""
import struct
import zlib

import numpy as np
import pytest

from wanderai.environment import Action, SceneSearchEnv
from wanderai.frontier_policy import FrontierPolicy
from wanderai.scene import default_scene
from wanderai.vision_policy import (PNG_SIGNATURE, VLMPolicy, VisionFrontierPolicy,
                                    build_vision_messages, encode_png,
                                    make_pixels_only_policy)


def _png_chunks(png: bytes) -> dict:
    """Walk the chunk stream: {tag: concatenated payload}."""
    assert png.startswith(PNG_SIGNATURE)
    pos, out = len(PNG_SIGNATURE), {}
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos:pos + 4])
        tag = png[pos + 4:pos + 8]
        out[tag] = out.get(tag, b"") + png[pos + 8:pos + 8 + length]
        pos += 12 + length          # length + tag + payload + crc
    return out


def test_encode_png_structure_and_roundtrip():
    h, w = 7, 5
    rgb = (np.arange(h * w * 3, dtype=np.uint32) % 256).astype(np.uint8).reshape(h, w, 3)
    png = encode_png(rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    for tag in (b"IHDR", b"IDAT", b"IEND"):
        assert tag in png
    chunks = _png_chunks(png)
    width, height, depth, color = struct.unpack(">IIBB", chunks[b"IHDR"][:10])
    assert (width, height, depth, color) == (w, h, 8, 2)
    raw = zlib.decompress(chunks[b"IDAT"])
    assert len(raw) == h * (1 + w * 3)      # filter byte + RGB row, per scanline
    # Filter type 0 everywhere -> pixel data survives byte-for-byte.
    rows = [raw[y * (1 + w * 3): (y + 1) * (1 + w * 3)] for y in range(h)]
    assert all(r[0] == 0 for r in rows)
    assert b"".join(r[1:] for r in rows) == rgb.tobytes()


def test_encode_png_rejects_non_uint8():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), dtype=np.uint8))


def test_build_vision_messages_embeds_image_and_scaffold():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    msgs = build_vision_messages(rgb, "Recent moves: none.")
    assert msgs[0]["role"] == "system"
    image, text = msgs[1]["content"]
    assert image["type"] == "image_url"
    assert image["image_url"]["url"].startswith("data:image/png;base64,")
    assert text["type"] == "text"
    assert "Recent moves: none." in text["text"]


def test_vlm_policy_acts_from_stub_transport():
    env = SceneSearchEnv(default_scene())
    env.reset()
    seen = []

    def transport(messages):
        seen.append(messages)
        return "I will go. ACTION=MOVE_FORWARD"

    pol = VLMPolicy(transport=transport)
    assert pol.act(None, env) == Action.MOVE_FORWARD
    assert pol.last_error is None
    # The transport received an image message (the policy really sends pixels).
    content = seen[0][1]["content"]
    assert any(part.get("type") == "image_url" for part in content)


def test_vlm_policy_falls_back_to_turn_left_on_error():
    env = SceneSearchEnv(default_scene())
    env.reset()

    def boom(messages):
        raise RuntimeError("network down")

    pol = VLMPolicy(transport=boom)
    assert pol.act(None, env) == Action.TURN_LEFT       # deterministic scan
    assert pol.last_error is not None
    assert "network down" in pol.last_error


def test_messages_are_deterministic_for_same_env_state():
    env = SceneSearchEnv(default_scene())
    env.reset()
    pol = VLMPolicy(transport=lambda m: "ACTION=TURN_RIGHT")
    assert pol.messages_for(env) == pol.messages_for(env)


def test_pixels_only_factory_is_a_frontier_policy():
    pol = make_pixels_only_policy()
    assert isinstance(pol, FrontierPolicy)
    assert isinstance(pol, VisionFrontierPolicy)
