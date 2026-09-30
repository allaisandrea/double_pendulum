import numpy as np
import pytest

from common.testing_lib import pose, write_frames
from imagination.ranking_lib import (Scores, agreement, agreements, episodes, exploitation_gap, mmrv,
                                     read_scores, subsets, write_scores)

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
    assert agreement(real, real - np.array([0.2, 0, 0, 0]))["bias"] == pytest.approx(-0.05)


def test_subsets_split_off_a_world_models_own_policies():
    seen = np.array([True, False, False])
    trained_in = ["wm3", "k8", "k8"]
    s = subsets("k8", seen, trained_in)
    assert s["all"].tolist() == [True] * 3 and s["new"].tolist() == [False, True, True]
    assert s["own"].tolist() == [False, True, True] and s["held_out"].tolist() == [True, False, False]
    assert set(subsets("k1", seen, trained_in)) == {"all", "new"}


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
    assert [round(e.real, 3) for e in episodes(path, STRAIGHT, history=17, seconds=5)] == [3.0, -3.0]
    assert eps[0].start.obs.shape == (1, 17, 3, 2)
    # The start window ends just before the drive at 15 s: still hanging.
    assert np.allclose(eps[0].start.obs[0, -1, :, 1].numpy(), -1.0)


def scores() -> Scores:
    rig = np.array([0.2, 0.5, 1.0, 1.2])
    return Scores(["a", "b", "c", "d"], ["old", "k8", "k8", "k8"], np.array([True, False, False, False]),
                  np.array([14, 14, 7, 14]), rig, np.array([0.01, 0.02, 0.03, 0.04]),
                  {"k8": rig + np.array([0.0, 0.5, 1.0, 1.5]), "k1": rig + 0.1})


def test_scores_round_trip(tmp_path):
    s = scores()
    write_scores(tmp_path / "scores.csv", s)
    back = read_scores(tmp_path / "scores.csv")
    assert back.policy == s.policy and back.trained_in == s.trained_in
    assert back.seen.tolist() == s.seen.tolist() and back.episodes.tolist() == s.episodes.tolist()
    np.testing.assert_allclose(back.rig, s.rig)
    assert list(back.predicted) == ["k8", "k1"]
    np.testing.assert_allclose(back.predicted["k8"], s.predicted["k8"])


def test_agreements_cover_the_subsets_of_three_or_more_and_the_gap_the_own_policies():
    s = scores()
    rows = {(a.world_model, a.policies): a for a in agreements(s)}
    # k8 has 3 own policies but 1 held out, too few to score.
    assert set(rows) == {("k8", "all"), ("k8", "new"), ("k8", "own"), ("k1", "all"), ("k1", "new")}
    assert rows["k8", "own"].n == 3 and rows["k8", "own"].metrics["bias"] == pytest.approx(1.0)
    assert exploitation_gap(s, "k8") == pytest.approx(1.0 - 0.0)
    assert exploitation_gap(s, "k1") is None


def test_a_world_model_spec_may_carry_its_own_tau():
    from pathlib import Path

    from imagination.ranking import world_model_spec

    assert world_model_spec("runs/world_model/stoch-b05/model.pt@0", 1.0) == (
        "stoch-b05@0", Path("runs/world_model/stoch-b05/model.pt"), 0.0)
    assert world_model_spec("runs/world_model/wm3/model.pt", 1.0) == ("wm3", Path("runs/world_model/wm3/model.pt"), 1.0)


def test_episodes_follow_the_duty_cycle_the_recording_says(tmp_path):
    # 30 s on, 8 s off, for 120 s: drives at 38 and 76 s (the one at 114 s
    # does not finish). Arms up during the drive at 38 s only.
    import pyarrow as pa, pyarrow.ipc as ipc
    n = 120 * 125
    t = np.arange(n) * 0.008
    up = (t >= 38) & (t < 68)
    path = tmp_path / "frames.arrows"
    write_frames(path, list(range(n)), [[pose(0.0 if u else np.pi)] * 3 for u in up], [0] * n)
    with ipc.open_stream(path) as f:
        table = f.read_all()
    table = table.append_column("t_capture", pa.array((t * 1e9).astype("int64")).cast(pa.duration("ns")))
    table = table.replace_schema_metadata({"active_ns": str(30 * 10**9), "rest_ns": str(8 * 10**9)})
    with ipc.new_stream(path, table.schema) as w:
        w.write_table(table)
    from imagination.ranking_lib import duty
    assert duty(path) == (30.0, 8.0)
    assert [round(e.real, 3) for e in episodes(path, STRAIGHT, history=17)] == [3.0, -3.0]
