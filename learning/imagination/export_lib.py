"""Exports a policy for the harness, which runs its forward pass in Rust.

The export is JSON: the actor's layers (tanh between them, none after the
last), the int8 action for each bin, and the policy window. With them come
test cases: histories of frames as the harness sees them, newest first,
with poses as quaternions, missed tags and dropped frames, and the logits
PyTorch gives for each. The harness recomputes every case when it loads
the policy and refuses it if any differs, so the export checks the whole
path from raw poses to logits, not just the arithmetic.

The cases go through the training pipeline itself: each history is
written as a frames table and read back with `load_recording`, so the
Rust side must match what the policy was trained on, gap filling and
action alignment included.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from common.data_lib import NUM_TAGS, load_recording
from common.testing_lib import write_frames
from imagination.agent_lib import Agent
from world_model.model_lib import step_features

FORMAT = "pendulum-policy-v1"


def actor_layers(agent: Agent) -> list[dict]:
    """Each linear layer of the actor: weight [out][in] and bias [out]."""
    linears = [m for m in agent.actor if isinstance(m, torch.nn.Linear)]
    return [{"weight": l.weight.tolist(), "bias": l.bias.tolist()} for l in linears]


def random_history(window: int, levels, rng: np.random.Generator) -> list[dict]:
    """A history of `window` + 1 frames and a few more, newest first, as
    the harness keeps it: some frames dropped, some tags missed, poses
    tilted a little out of the image plane, as the detector reports them."""
    frames, frame = [], 1000
    for _ in range(window + 4):
        frames.append(frame)
        frame += 1 + (rng.random() < 0.1) * int(rng.integers(1, 3))
    steps = []
    for i, f in enumerate(frames):
        poses = []
        for _ in range(NUM_TAGS):
            if rng.random() < 0.15:
                poses.append(None)
                continue
            q = np.array([1.0, *rng.normal(0, 0.1, 2), 0.0])
            yaw = rng.uniform(-np.pi, np.pi)
            q = quaternion_product(np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]), q)
            q /= np.linalg.norm(q)
            q *= np.sign(q[0]) or 1.0
            poses.append([*rng.normal(0, 0.1, 2), 0.7, *q])
        last = i == len(frames) - 1
        steps.append({"frame": f, "poses": poses, "action": None if last else int(rng.choice(levels))})
    return steps[::-1]


def quaternion_product(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


@torch.no_grad()
def reference_logits(agent: Agent, window: int, history: list[dict]) -> list[float]:
    """The logits for `history` (newest first), by way of a frames table
    read back as the training data is."""
    oldest_first = history[::-1]
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "frames.arrows"
        write_frames(
            path,
            [s["frame"] for s in oldest_first],
            [s["poses"] for s in oldest_first],
            # The newest frame's action is the one being chosen; its value
            # never reaches the policy.
            [s["action"] if s["action"] is not None else 0 for s in oldest_first],
        )
        r = load_recording(path)
    obs, present, action = (torch.from_numpy(a) for a in (r.obs, r.present, r.action))
    # As ImaginedEnv.observe: the latest frames, each with the action before it.
    x = step_features(obs[None, -window:], present[None, -window:], action[None, -window - 1 : -1])
    return agent.actor(x.flatten(1).float())[0].tolist()


def export_policy(agent: Agent, levels, window: int, path: Path, source: dict, cases: int = 16, seed: int = 0):
    """Writes the JSON export of `agent` to `path`."""
    agent = agent.cpu().eval()
    rng = np.random.default_rng(seed)
    levels = [int(l) for l in levels]
    tests = []
    for _ in range(cases):
        history = random_history(window, levels, rng)
        logits = reference_logits(agent, window, history)
        tests.append({"history": history, "logits": logits, "action": levels[int(np.argmax(logits))]})
    export = {
        "format": FORMAT,
        "source": source,
        "policy_window": window,
        "levels": levels,
        "layers": actor_layers(agent),
        "cases": tests,
    }
    Path(path).write_text(json.dumps(export))
