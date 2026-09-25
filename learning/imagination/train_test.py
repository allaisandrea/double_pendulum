import numpy as np
import pytest
import torch

from common.testing_lib import pose, write_frames
from imagination.train import main
from world_model.model_lib import WorldModel

CONFIG = """
world_model = "wm.pt"
data_dir = "data"
recordings = ["r"]
action_step = 16
action_bins = 9
policy_window = 4
sample_missing = true
hidden = 8
layers = 1
num_envs = 6
rollout_steps = 5
reset_horizons = 0.5
iterations = 6
update_epochs = 2
minibatches = 2
lr = 1e-3
warmup_iterations = 1
cooldown_fraction = 0.3
gamma = 0.9
gae_lambda = 0.9
clip = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
seed = 0
eval_every = 2
eval_steps = 40
eval_seed = 0
checkpoint_every = 3
wandb_project = "test"
"""


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A directory with a config, a small world model and one recording
    that ends in a long rest."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "base.toml").write_text(CONFIG)
    torch.manual_seed(0)
    model = WorldModel(window=3, hidden=8, layers=1, delta_scale=0.05)
    torch.save(
        {"config": {"window": 3, "hidden": 8, "layers": 1}, "delta_scale": 0.05, "model": model.state_dict()},
        "wm.pt",
    )
    (tmp_path / "data" / "r").mkdir(parents=True)
    rng = np.random.default_rng(0)
    driven = [[pose(a) for a in rng.uniform(-np.pi, np.pi, 3)] for _ in range(100)]
    rest = [[pose(np.pi)] * 3] * 600
    write_frames(tmp_path / "data" / "r" / "frames.arrows", list(range(700)), driven + rest, [30] * 100 + [0] * 600)
    return tmp_path


def agent_state(path):
    return torch.load(path, weights_only=False)["agent"]


def test_resuming_from_a_checkpoint_repeats_the_run_exactly(workdir):
    common = ["--wandb", "disabled", "--device", "cpu"]
    main(["base.toml", "--name", "straight", *common])
    runs = workdir / "runs" / "policy"
    assert sorted(p.name for p in (runs / "straight" / "checkpoints").iterdir()) == [
        "iter_000003.pt",
        "iter_000006.pt",
    ]
    assert (runs / "straight" / "rollout.mp4").exists()

    main(["base.toml", "--name", "resumed", "--iterations", "6", *common])
    # Drop the second half and resume it from the first checkpoint.
    (runs / "resumed" / "checkpoints" / "iter_000006.pt").unlink()
    main(["--resume", str(runs / "resumed" / "checkpoints" / "iter_000003.pt"), *common])

    straight = agent_state(runs / "straight" / "policy.pt")
    resumed = agent_state(runs / "resumed" / "policy.pt")
    assert all(torch.equal(straight[k], resumed[k]) for k in straight)
    assert torch.load(runs / "resumed" / "policy.pt", weights_only=False)["iteration"] == 6


def test_a_new_run_needs_a_name_and_a_resume_takes_nothing_else(workdir):
    with pytest.raises(SystemExit):
        main(["base.toml", "--wandb", "disabled"])
    with pytest.raises(SystemExit):
        main(["base.toml", "--resume", "x.pt", "--wandb", "disabled"])
