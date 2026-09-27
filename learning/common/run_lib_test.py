import pytest

from common.run_lib import latest_checkpoint, load_config


def test_load_config_applies_overrides_to_known_and_optional_keys(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('lr = 1e-3\nname = "a"\n')
    cfg = load_config(path, ["lr=2e-3", "compile=true"], optional={"compile"})
    assert cfg == {"lr": 2e-3, "name": "a", "compile": True}
    with pytest.raises(SystemExit):
        load_config(path, ["compile=true"])
    with pytest.raises(SystemExit):
        load_config(path, ["lr"])


def test_latest_checkpoint_is_the_highest_step(tmp_path):
    assert latest_checkpoint(tmp_path) is None
    (tmp_path / "checkpoints").mkdir()
    for step in (900, 10000, 2000):
        (tmp_path / "checkpoints" / f"step_{step:07d}.pt").touch()
    assert latest_checkpoint(tmp_path).name == "step_0010000.pt"
