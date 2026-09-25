import numpy as np
import pytest
import torch

from common.data_lib import ACTION_SCALE, Windows, correct_yaws, hanging_yaws, load_recording, yaw
from common.testing_lib import pose, write_frames


def test_yaw_reads_the_in_plane_angle():
    assert yaw(np.array(pose(0.3))) == pytest.approx(0.3, abs=1e-6)
    assert yaw(np.array(pose(-2.5))) == pytest.approx(-2.5, abs=1e-6)


def test_a_dropped_frame_is_unseen_and_carries_the_action(tmp_path):
    path = tmp_path / "frames.arrows"
    write_frames(
        path,
        frames=[10, 11, 14],
        poses=[[pose(0.1), None, pose(0.2)]] * 3,
        actions=[-128, 40, 127],
    )
    r = load_recording(path)
    assert r.present.tolist() == [[True, False, True]] * 2 + [[False] * 3] * 2 + [[True, False, True]]
    assert (r.action * ACTION_SCALE).tolist() == [-128, 40, 40, 40, 127]
    assert r.obs[0, 0] == pytest.approx([np.sin(0.1), np.cos(0.1)], abs=1e-6)
    assert r.obs[0, 1].tolist() == [0, 0]


def test_windows_do_not_span_recordings(tmp_path):
    recs = []
    for i, n in enumerate([5, 3]):
        p = tmp_path / f"{i}.arrows"
        write_frames(p, list(range(n)), [[pose(0.0)] * 3] * n, [i] * n)
        recs.append(load_recording(p))
    w = Windows(recs, 3, "cpu")
    assert len(w) == 3 + 1
    batch = w.gather(torch.arange(4))
    assert (batch.action * ACTION_SCALE)[:, 0].tolist() == [0, 0, 0, 1]




def test_hanging_yaws_average_the_ends_of_long_rests(tmp_path):
    path = tmp_path / "frames.arrows"
    # Rests (action 0) at yaws either side of 180°, and a driven stretch
    # at another yaw that must not count.
    rest = [[pose(np.pi - 0.1), pose(np.pi + 0.1), pose(np.pi)]] * 130
    driven = [[pose(1.0)] * 3] * 20
    poses = driven + rest + driven
    actions = [30] * 20 + [0] * 130 + [30] * 20
    write_frames(path, list(range(len(actions))), poses, actions)
    h = hanging_yaws([load_recording(path)], min_rest=100).numpy()
    angles = np.arctan2(h[:, 0], h[:, 1])
    assert np.abs(angles) == pytest.approx([np.pi - 0.1, np.pi - 0.1, np.pi], abs=1e-5)
    with pytest.raises(ValueError):
        hanging_yaws([load_recording(path)], min_rest=200)


def test_corrected_yaws_hang_at_180_and_keep_their_differences():
    offsets = np.array([-0.1, 0.05, 0.3])
    hanging = np.stack([np.sin(np.pi + offsets), np.cos(np.pi + offsets)], -1)
    yaw = np.array([[np.pi, 0.0, 1.0]]) + offsets
    obs = np.stack([np.sin(yaw), np.cos(yaw)], -1)
    fixed = correct_yaws(obs, hanging)
    angles = np.arctan2(fixed[..., 0], fixed[..., 1])
    assert np.abs(angles[0, 0]) == pytest.approx(np.pi)
    assert angles[0, 1:] == pytest.approx([0.0, 1.0])
    fixed_t = correct_yaws(torch.from_numpy(obs), torch.from_numpy(hanging))
    assert fixed_t.numpy() == pytest.approx(fixed)
