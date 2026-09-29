"""Regenerates the CSV files in docs/results that docs/figures.py plots,
from the sources the experiments used, from learning/:

    uv run python -m docs.results            # every source
    uv run python -m docs.results ranking    # some of them

wandb: world_model.report of the batch sweep, the scaling runs, the
rollout runs and the stochastic scaling runs (batch.csv, scaling.csv,
rollout.csv), and the stochastic scaling study (stochastic_scaling.csv,
stochastic_diagnosis.csv, stochastic_scaling_clip.csv, two_stage_joint.csv,
two_stage.csv), from W&B.
profile: world_model.profile timed here (profile_<device>.csv, on the Mac
profile_mps.csv), and the logs of the cloud profiling jobs, fetched from
S3 (profile_cloud.csv).
sim2real: imagination.sim2real of the rollout-trained models (sim2real.csv).
stochastic: world_model.compare of the stochastic world models against
the one-step one, with their ensembles (stochastic.csv,
stochastic_upright.csv).
ranking: imagination.ranking imagine of the three ranking sessions of
2026-09-27 (ranking_scores.csv), then agreement.
agreement: imagination.ranking agreement of ranking_scores.csv
(ranking_agreement.csv), without imagining again; ranking ends with it,
so only a change of metrics needs it alone.

anneal: imagination.ranking imagine of every greedy rig test in the 256 x 3
world model, sampled and on its mean, and in stoch-it5 (s256_scores.csv,
s256_agreement.csv), and each policy's time upright and largest actions
on the rig (rig_stats.csv).
loop: world_model.compare of the loop's world models on each collection's
validation recording (loop_world_models.csv, loop_world_models_upright.csv),
and imagination.ranking imagine of its policies' greedy rig tests
(loop_scores.csv).

sim2real, stochastic, ranking and loop need the world models and policies in runs/ and the
recordings in ../recordings; profile needs AWS credentials (profile
andrea-personal).
"""
import argparse
import subprocess
import sys
from pathlib import Path

from world_model.profile import parse_log, write_csv

RESULTS = Path(__file__).parent / "results"
S3_RUNS = "s3://allais-andrea-store/double_pendulum/runs/run"
# The cloud jobs that ran world_model.profile: an L4 (g6.xlarge) and a T4 (g4dn.xlarge).
CLOUD_PROFILES = ["profile-od-g6", "profile-spot"]
PROFILE_GRID = ["--widths", "256", "512", "1024", "--depths", "3", "5", "--batches", "1024", "4096", "16384"]
# The world models compared in closed loop, by training rollout length.
WORLD_MODELS = ["wm3", "wm-w512-d3-cd50k", "rollout-k1", "rollout-k4", "rollout3-k6", "rollout3-k8",
                "rollout3-k8-seed1", "rollout3-k10", "rollout3-k12", "rollout-k16", "rollout2-k32",
                "rollout3-k64-clip"]
# The stochastic world models, in the ranking sampled (tau 1, as policies
# train in them) and, for the first, on its mean (@0).
STOCHASTIC = ["stoch-b05", "stoch-b0"]
# The ranking sessions' manifests: session 1 (2026-09-27 morning, 13
# policies), session 2 (afternoon, the K = 1 and K = 6 candidates and
# ppo-wm3; its first pass split by camera drops) and session 3 (ppo-stoch).
RANKING_SESSIONS = [f"../recordings/ranking-{m}.tsv" for m in [
    "1790523610-seed0", "1790525401-seed1",
    "1790541846-seed0", "1790543406-seed0", "1790544933-seed0", "1790545071-seed1",
    "1790548265-seed0", "1790548404-seed1"]]
# The world model / policy / data loop of 2026-09-28: collection k's
# 5-minute validation recording, its policy, and the world model trained
# with it first; the greedy rig tests of each iteration's policy.
LOOP = [("1790611634", "ppo-stoch, with bursts", "stoch-it1"),
        ("1790619211", "ppo-stoch-it1", "stoch-it2"),
        ("1790626779", "ppo-stoch-it2", "stoch-it3")]
LOOP_MODELS = ["stoch-b05", "stoch-it1", "stoch-it2", "stoch-it3"]
LOOP_GREEDY = [f"../recordings/ranking-{m}.tsv" for m in [
    "1790548265-seed0", "1790548404-seed1",  # ppo-stoch, session 3
    "1790618903-seed0", "1790619056-seed1",  # ppo-stoch-it1
    "1790626471-seed0", "1790626625-seed1",  # ppo-stoch-it2
    "1790634883-seed0", "1790635036-seed1",  # ppo-stoch-it3
]]
# Greedy rig tests of the 256 x 3 model's policies (2026-09-29): the one
# trained at tau 1, at tau 0, the anneal from tau 0's iteration 3000 (and its
# iteration 1750), and the anneals from scratch over 5,000 and 8,000.
S256_GREEDY = [f"../recordings/ranking-{m}.tsv" for m in [
    "1790712456-seed0", "1790712595-seed1",  # ppo-s256-anneal@1750
    "1790712851-seed0", "1790712989-seed1",  # ppo-s256-mean
    "1790713181-seed0", "1790713319-seed1",  # ppo-s256
    "1790713681-seed0", "1790713820-seed1",  # ppo-s256-anneal
    "1790714289-seed0", "1790714427-seed1",  # ppo-s256-anneal0-5k
    "1790714695-seed0", "1790714833-seed1",  # ppo-s256-anneal0-8k
]]
# The loop's fourth policy's greedy test.
IT4_GREEDY = ["../recordings/ranking-1790644654-seed0.tsv", "../recordings/ranking-1790644807-seed1.tsv"]
# The validation recordings the stochastic models are compared on, as
# world_model.train names them: val_policy_wm3 and val_random_walk.
STOCHASTIC_VAL = ["1790464558", "1790281442"]


def run(*command: str):
    print("$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def python(module: str, *args: str):
    run(sys.executable, "-m", module, *args)


def wandb():
    python("world_model.report", "batch-", "--sets", "val_policy_wm3", "--horizons", "16", "64",
           "--csv", str(RESULTS / "batch.csv"))
    python("world_model.report", "wm-w", "--sets", "val_policy_wm3", "val_random_walk",
           "--horizons", "16", "64", "125", "--csv", str(RESULTS / "scaling.csv"))
    python("world_model.report", "rollout", "--sets", "val_policy_wm3", "val_random_walk",
           "--horizons", "16", "64", "125", "--csv", str(RESULTS / "rollout.csv"))
    sets = ["val_all", "val_random_walk", "val_policy_wm3", "val_ppo_stoch_it4", "train"]
    # The stochastic scaling study: the first grid (diverged), the diagnosis,
    # the grid again with gradient clipping, and the two-stage comparison.
    for prefix, name in [("sw-w", "stochastic_scaling"), ("sdiag-", "stochastic_diagnosis"),
                         ("sc-w", "stochastic_scaling_clip"), ("sj-", "two_stage_joint"), ("sv-", "two_stage")]:
        python("world_model.report", prefix, "--sets", *sets, "--all", "--csv", str(RESULTS / f"{name}.csv"))


def profile():
    import torch
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    python("world_model.profile", *PROFILE_GRID, "--csv", str(RESULTS / f"profile_{device}.csv"))
    rows = []
    for job in CLOUD_PROFILES:
        log = subprocess.run(["aws", "--profile", "andrea-personal", "s3", "cp", f"{S3_RUNS}/{job}/job.log", "-"],
                             check=True, capture_output=True, text=True).stdout
        rows += parse_log(log)
    write_csv(RESULTS / "profile_cloud.csv", rows)


def sim2real():
    python("imagination.sim2real", *(f"runs/world_model/{m}/model.pt" for m in WORLD_MODELS if m != "wm-w512-d3-cd50k"),
           "--csv", str(RESULTS / "sim2real.csv"))


def ranking():
    python("imagination.ranking", "imagine", *RANKING_SESSIONS,
           "--world-models", *(f"runs/world_model/{m}/model.pt" for m in WORLD_MODELS + STOCHASTIC),
           f"runs/world_model/{STOCHASTIC[0]}/model.pt@0",
           "--scores", str(RESULTS / "ranking_scores.csv"))
    agreement()


def stochastic():
    """world_model.compare of the stochastic models and the one-step model
    they share a recipe with, including their ensembles, on all windows and
    on those starting upright."""
    models = [f"runs/world_model/{m}/model.pt" for m in ["rollout-k1", *STOCHASTIC]]
    python("world_model.compare", *models, "--val", *STOCHASTIC_VAL, "--csv", str(RESULTS / "stochastic.csv"))
    python("world_model.compare", *models, "--val", *STOCHASTIC_VAL, "--upright",
           "--csv", str(RESULTS / "stochastic_upright.csv"))


def agreement():
    python("imagination.ranking", "agreement", str(RESULTS / "ranking_scores.csv"),
           "--csv", str(RESULTS / "ranking_agreement.csv"))


def loop():
    """The loop's world models on every collection's validation recording,
    and its policies' greedy rig tests scored in each of them."""
    models = [f"runs/world_model/{m}/model.pt" for m in LOOP_MODELS]
    vals = [v for v, _, _ in LOOP] + STOCHASTIC_VAL
    python("world_model.compare", *models, "--val", *vals, "--csv", str(RESULTS / "loop_world_models.csv"))
    python("world_model.compare", *models, "--val", *vals, "--upright",
           "--csv", str(RESULTS / "loop_world_models_upright.csv"))
    python("imagination.ranking", "imagine", *LOOP_GREEDY, "--world-models", *models,
           "--scores", str(RESULTS / "loop_scores.csv"))


def anneal():
    """Every greedy rig test, scored in the 256 x 3 world model sampled and
    on its mean (and the 512 x 3 trained on the same data), and each
    policy's time with each arm up and its use of the largest actions."""
    manifests = RANKING_SESSIONS + LOOP_GREEDY[2:] + IT4_GREEDY + S256_GREEDY
    s256 = "runs/world_model/sc-w256-d3-cd200k/model.pt"
    python("imagination.ranking", "imagine", *manifests,
           "--world-models", s256, f"{s256}@0", "runs/world_model/stoch-it5/model.pt",
           "--scores", str(RESULTS / "s256_scores.csv"))
    python("imagination.ranking", "agreement", str(RESULTS / "s256_scores.csv"),
           "--csv", str(RESULTS / "s256_agreement.csv"))
    rig_stats(manifests, RESULTS / "rig_stats.csv")


def rig_stats(manifests: list[str], out: Path):
    """Per policy, over the ranking's drives (each after the first, 10 s):
    the share of frames with each arm within 30° of upright, all three,
    and the share of actions beyond ±64, ±96 and at ±126 or more."""
    import csv
    import numpy as np
    from common.data_lib import ACTION_SCALE, correct_yaws, hanging_yaws, load_recording, load_recordings
    from common.render_lib import fill_unseen
    from imagination.ranking_lib import capture_times
    from imagination.sim2real import HANGING

    hanging = hanging_yaws(load_recordings(Path("data"), HANGING)).numpy()
    up_cos = np.cos(np.radians(30))
    frames = {}
    for m in manifests:
        for line in Path(m).read_text().splitlines():
            name, rec = line.split("\t")
            f = Path(rec) / "frames.arrows"
            r = load_recording(f)
            at = capture_times(f, len(r.action))
            drive = ((at % 15) < 10) & (at >= 15)
            cos = correct_yaws(fill_unseen(r.obs, r.present), hanging)[..., 1]
            a, up = np.round(r.action * ACTION_SCALE)[drive], (cos > up_cos)[drive]
            frames.setdefault(name, []).append((a, up))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "arm0_up", "arm1_up", "arm2_up", "all_up", "beyond_64", "beyond_96", "at_126"])
        for name, parts in frames.items():
            a = np.concatenate([p[0] for p in parts])
            up = np.concatenate([p[1] for p in parts])
            w.writerow([name, *(round(float(x), 4) for x in up.mean(0)), round(float(up.all(-1).mean()), 4),
                        *(round(float(x), 4) for x in ((np.abs(a) > 64).mean(), (np.abs(a) > 96).mean(),
                                                         (np.abs(a) >= 126).mean()))])


STEPS = {"wandb": wandb, "profile": profile, "sim2real": sim2real, "stochastic": stochastic, "ranking": ranking,
         "agreement": agreement, "loop": loop, "anneal": anneal}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("steps", nargs="*", help=f"any of {', '.join(STEPS)}; all by default")
    args = parser.parse_args()
    unknown = set(args.steps) - set(STEPS)
    if unknown:
        parser.error(f"unknown steps: {', '.join(sorted(unknown))}")
    RESULTS.mkdir(exist_ok=True)
    for step in args.steps or [s for s in STEPS if s != "agreement"]:
        STEPS[step]()


if __name__ == "__main__":
    main()
