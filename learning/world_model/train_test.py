import numpy as np
import pytest
import torch

from common.testing_lib import pose, write_frames
from world_model.train import main

CONFIG = """
data_dir = "data"
train = ["t"]
window = 4
hidden = 8
layers = 1
batch_size = 16
lr = 1e-3
warmup_steps = 2
cooldown_fraction = 0.2
weight_decay = 1e-4
steps = 10
bce_weight = 0.1
seed = 0
log_every = 5
eval_every = 5
checkpoint_every = 5
eval_samples = 32
horizons = [1, 4]
wandb_project = "test"

[val]
v = ["v"]
"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A config and two recordings, each with a long rest at the end."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base.toml").write_text(CONFIG)
    rng = np.random.default_rng(0)
    for name in ["t", "v"]:
        (tmp_path / "data" / name).mkdir(parents=True)
        driven = [[pose(a) for a in rng.uniform(-np.pi, np.pi, 3)] for _ in range(300)]
        rest = [[pose(np.pi)] * 3] * 600
        write_frames(tmp_path / "data" / name / "frames.arrows", list(range(900)), driven + rest,
                     list(rng.integers(-60, 61, 300)) + [0] * 600)
    return tmp_path


COMMON = ["--wandb", "disabled", "--device", "cpu"]


def weights(path):
    return torch.load(path, weights_only=False)["model"]


def test_resuming_from_a_checkpoint_repeats_the_run_exactly(workdir):
    main(["base.toml", "--name", "straight", *COMMON])
    main(["base.toml", "--name", "resumed", *COMMON])
    runs = workdir / "runs" / "world_model"
    assert sorted(p.name for p in (runs / "straight" / "checkpoints").iterdir()) == [
        "step_0000005.pt", "step_0000010.pt"]
    (runs / "resumed" / "checkpoints" / "step_0000010.pt").unlink()
    main(["--resume", str(runs / "resumed" / "checkpoints" / "step_0000005.pt"), *COMMON])
    a, b = weights(runs / "straight" / "model.pt"), weights(runs / "resumed" / "model.pt")
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_a_branch_cools_down_from_its_parents_checkpoint(workdir):
    main(["base.toml", "--name", "long", "--set", "cooldown_fraction=0", *COMMON])
    runs = workdir / "runs" / "world_model"
    parent = torch.load(runs / "long" / "checkpoints" / "step_0000005.pt", weights_only=False)
    # With no cooldown, the rate holds at lr after the warmup.
    assert parent["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-3)
    main(["--branch", str(runs / "long" / "checkpoints" / "step_0000005.pt"), "--cooldown-steps", "4",
          "--name", "long-cd5", *COMMON])
    branch = torch.load(runs / "long-cd5" / "checkpoints" / "step_0000009.pt", weights_only=False)
    assert branch["config"]["steps"] == 9 and branch["config"]["cooldown_steps"] == 4
    assert branch["config"]["branch_of"] == {"run": "long", "step": 5}
    # The last step's rate: 1 - sqrt(3/4) of lr.
    assert branch["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-3 * (1 - 0.75**0.5))


def test_set_overrides_and_unknown_keys_are_refused(workdir):
    main(["base.toml", "--name", "wide", "--set", "hidden=12", "--steps", "5", *COMMON])
    assert torch.load(workdir / "runs/world_model/wide/model.pt", weights_only=False)["config"]["hidden"] == 12
    with pytest.raises(SystemExit):
        main(["base.toml", "--name", "bad", "--set", "nope=1", *COMMON])
