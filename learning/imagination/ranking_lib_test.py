import numpy as np
import pytest

from common.testing_lib import pose, write_frames
from imagination.ranking_lib import agreement, episodes, mmrv

STRAIGHT = np.array([[0.0, -1.0]] * 3)


def test_mmrv_is_zero_for_the_same_order_and_weighs_swaps_by_the_real_gap():
    real = np.array([0.1, 0.5, 1.0, 2.0])
    assert mmrv(real, real * 3 - 1) == 0.0
    # Swapping the two near-equals costs little, swapping the ends a lot.
    near = mmrv(real, np.array([0.5, 0.1, 1.0, 2.0]))
    far = mmrv(real, np.array([2.0, 0.5, 1.0, 0.1]))
    assert near == pytest.approx(2 * 0.4 / 4)
    assert far > near
    a = agreement(real, real + 0.1)
    assert a["spearman"] == pytest.approx(1.0) and a["mae"] == pytest.approx(0.1)


def test_episodes_start_after_each_rest_and_score_the_drive(tmp_path):
    # 125 fps for 41 s on a 10 s on, 5 s off cycle: drives at 0, 15 and 30 s.
    # Every arm hangs (cos -1) except during the drive at 15 s, when all are
    # up (cos +1), so that episode scores +3 and the one at 30 s -3.
    n = 41 * 125
    t = np.arange(n) * 0.008
    up = (t >= 15) & (t < 25)
    poses = [[pose(0.0 if u else np.pi)] * 3 for u in up]
    path = tmp_path / "frames.arrows"
    write_frames(path, list(range(n)), poses, [0] * n)
    # write_frames leaves out capture times; add them as the rig records them.
    import pyarrow as pa, pyarrow.ipc as ipc
    with ipc.open_stream(path) as f:
        table = f.read_all()
    table = table.append_column("t_capture", pa.array((t * 1e9).astype("int64")).cast(pa.duration("ns")))
    with ipc.new_stream(path, table.schema) as w:
        w.write_table(table)

    eps = episodes(path, STRAIGHT, history=17)
    assert [round(e.real, 3) for e in eps] == [3.0, -3.0]
    assert eps[0].start.obs.shape == (1, 17, 3, 2)
    # The start window ends just before the drive at 15 s: still hanging.
    assert np.allclose(eps[0].start.obs[0, -1, :, 1].numpy(), -1.0)
