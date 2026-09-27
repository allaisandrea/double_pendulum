"""Compares world models' predictions of policies with what the policies did on the rig.

    uv run python -m imagination.sim2real runs/world_model/wm3/model.pt runs/world_model/rollout-k16/model.pt

The closer a world model predicts how a policy really performs, the
better it is to train policies in, whatever its one-step error. For each
policy tested greedily on the rig from hanging, this takes the mean sum
of cosines over the first --seconds of driving in its recording, and each
world model's prediction of it: the mean over --rollouts rollouts from
hanging, as imagination.evaluate runs them. Prints the table, and per
world model the mean absolute error, the correlation, and whether it
ranks the policies as the rig does.
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc

from common.data_lib import correct_yaws, hanging_yaws, load_recording, load_recordings
from common.render_lib import fill_unseen
from common.run_lib import pick_device
from imagination.evaluate import evaluate_policy
from imagination.rollout_lib import load_policy
from world_model.model_lib import load_world_model

# Greedy tests on the rig from hanging, without the speed governor.
TESTS = {
    "ppo-persistent": "1790370274",
    "ppo-speed": "1790372332",
    "ppo-wm2": "1790378236",
    "ppo-wm3": "1790454625",
}
# The recordings the hanging yaws are measured from, as policies were trained.
HANGING = ["1790282998", "1790279628", "1790281442", "1790372784", "1790378922"]


def real_cos_sum(recording: Path, hanging, seconds: float) -> float:
    r = load_recording(recording)
    obs = correct_yaws(fill_unseen(r.obs, r.present), hanging)
    with ipc.open_stream(recording) as f:
        t = f.read_all()
    capture = t["t_capture"].cast("int64").to_numpy() / 1e9
    frame = t["frame"].to_numpy()
    at = np.interp(np.arange(len(r.action)), frame - frame[0], capture)
    return float(obs[(at > 0) & (at < seconds)][..., 1].sum(-1).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("world_models", type=Path, nargs="+")
    parser.add_argument("--recordings", type=Path, default=Path("../recordings"))
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = pick_device(args.device)
    hanging = hanging_yaws(load_recordings(Path("data"), HANGING)).numpy()
    real = {p: real_cos_sum(args.recordings / r / "frames.arrows", hanging, args.seconds) for p, r in TESTS.items()}
    names = [p.parent.name for p in args.world_models]
    models = [load_world_model(p, device) for p in args.world_models]
    steps = round(args.seconds / 0.008)
    print(f"sum of cosines per frame, first {args.seconds:g} s from hanging")
    print(f"{'policy':16s} {'real':>7s} " + " ".join(f"{n:>18s}" for n in names))
    predicted = {n: [] for n in names}
    for policy in TESTS:
        row = []
        for name, model in zip(names, models):
            agent, env, _ = load_policy(f"runs/policy/{policy}/policy.pt", device)
            env.model = model
            v = evaluate_policy(agent, env, args.rollouts, steps, 0)["cos_sum"]
            predicted[name].append(v)
            row.append(v)
        print(f"{policy:16s} {real[policy]:+7.2f} " + " ".join(f"{v:+18.2f}" for v in row), flush=True)
    r = np.array(list(real.values()))
    for name in names:
        p = np.array(predicted[name])
        same = (np.argsort(p) == np.argsort(r)).all()
        print(f"{name}: mean |predicted - real| {np.abs(p - r).mean():.2f}, correlation {np.corrcoef(p, r)[0, 1]:+.2f}, "
              f"ranks the policies {'as the rig does' if same else 'differently'}")


if __name__ == "__main__":
    main()
