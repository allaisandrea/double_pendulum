import numpy as np
import pytest
import torch

from common.data_lib import ACTION_SCALE, Windows, load_recording, rest_window, yaw
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


def test_the_rest_window_ends_a_long_rest_with_every_tag_seen(tmp_path):
    path = tmp_path / "frames.arrows"
    seen = [pose(np.pi)] * 3
    poses = [seen] * 5 + [seen] * 7 + [[None] * 3] + [seen] * 3
    actions = [10] * 5 + [0] * 7 + [0] + [20] * 3
    write_frames(path, list(range(len(actions))), poses, actions)
    r = load_recording(path)
    # The rest runs over frames 5-12, but frame 12 misses every tag, so
    # the window ends at frame 11.
    batch = rest_window([r], 4, min_rest=6)
    assert batch.obs.shape == (1, 4, 3, 2)
    assert np.array_equal(batch.obs[0].numpy(), r.obs[8:12])
    assert batch.present.all()
    assert (batch.action == 0).all()
    with pytest.raises(ValueError):
        rest_window([r], 4, min_rest=10)
