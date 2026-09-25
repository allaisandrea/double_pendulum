import imageio.v3 as iio
import numpy as np

from common.render_lib import ARMS, SIZE, SURFACE, _fade, fill_unseen, render, render_frame, write_video


def hanging(n=1):
    return np.tile(np.array([[0.0, -1.0]] * 3, np.float32), (n, 1, 1))


def colour_below_pivot(frame, arm):
    """The pixel halfway down arm `arm` of a hanging pendulum."""
    x, y = SIZE // 2, int(SIZE * 0.46 + SIZE * 0.13 * (arm + 0.5))
    return tuple(frame[y, x])


def test_a_hanging_pendulum_is_drawn_straight_down():
    frame = render_frame(hanging()[0], [True] * 3, 0.0, 0.0)
    assert frame.shape == (SIZE, SIZE, 3) and frame.dtype == np.uint8
    for arm in range(3):
        assert colour_below_pivot(frame, arm) == ARMS[arm]
    # Nothing above the pivot.
    assert tuple(frame[int(SIZE * 0.46 - SIZE * 0.2), SIZE // 2]) == SURFACE


def test_an_unseen_arm_is_faded():
    frame = render_frame(hanging()[0], [True, False, True], 0.0, 0.0)
    assert colour_below_pivot(frame, 1) == _fade(ARMS[1])


def test_unseen_tags_keep_their_last_seen_angle():
    obs = np.zeros((3, 3, 2), np.float32)
    obs[0, 0] = [1.0, 0.0]
    present = np.array([[True, False, False]] + [[False] * 3] * 2)
    filled = fill_unseen(obs, present)
    assert filled[2, 0].tolist() == [1.0, 0.0]
    assert filled[2, 1].tolist() == [0.0, -1.0]


def test_a_video_has_one_frame_per_stride(tmp_path):
    frames = render(hanging(10), np.ones((10, 3), bool), np.zeros(10), stride=2)
    assert len(frames) == 5
    path = tmp_path / "v.mp4"
    write_video(path, frames)
    assert iio.imread(path).shape[0] == 5
