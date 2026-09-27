"""Regenerates the CSV files in docs/results that docs/figures.py plots,
from the sources the experiments used, from learning/:

    uv run python -m docs.results            # every source
    uv run python -m docs.results ranking    # some of them

wandb: world_model.report of the batch sweep, the scaling runs and the
rollout runs (batch.csv, scaling.csv, rollout.csv), from W&B.
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

sim2real, stochastic and ranking need the world models and policies in runs/ and the
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


STEPS = {"wandb": wandb, "profile": profile, "sim2real": sim2real, "stochastic": stochastic, "ranking": ranking,
         "agreement": agreement}


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
