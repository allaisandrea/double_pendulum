#!/bin/bash
# Runs one training job on an EC2 instance, then shuts it down, which
# terminates it. launch.py starts it from the instance's user data, in the
# unpacked code, as:
#
#   job.sh S3_ROOT KIND NAME CONFIG [ARGS...]
#
# S3_ROOT holds data/, runs/ and code/ (s3://allais-andrea-store/double_pendulum).
# KIND is world_model or policy, trained with CONFIG and ARGS as
# world_model.train or imagination.train would be; bench, which times 100
# policy iterations and 5000 world model steps without W&B; or run, which
# runs the module CONFIG names with ARGS (python -m CONFIG ARGS); or scale,
# which trains world_model.scale's long run NAME and its branches NAME-cd*
# with CONFIG and ARGS, carrying on from whatever of them S3 holds; or
# sweep, which runs world_model.sweep on the file CONFIG, whose runs must
# be named NAME-*.
#
# The run directory and this job's log go to S3_ROOT/runs/KIND/NAME every
# 5 minutes and at the end, with job_status holding the exit status. A
# policy job that finds checkpoints of NAME there resumes from the latest,
# so launching it again after a spot interruption carries on. A world
# model job has no resume: it starts over, replacing what an interrupted
# one left.
set -uo pipefail
export HOME=${HOME:-/root}
S3_ROOT=$1 KIND=$2 NAME=$3 CONFIG=$4
shift 4
ARGS=("$@")
LOG=/var/log/job.log
exec > >(tee -a "$LOG") 2>&1
cd "$(dirname "$0")/.." || exit 1  # learning/
RUN_DIR=runs/$KIND/$NAME
RUN_URI=$S3_ROOT/runs/$KIND/$NAME
# A scale job's runs: world model runs named NAME and NAME-cd*.
[ "$KIND" = scale ] || [ "$KIND" = sweep ] && RUN_URI=$S3_ROOT/runs/world_model/$NAME
SCALE_SYNC=(--exclude "*" --include "$NAME/*" --include "$NAME-*")

upload() {
    if [ "$KIND" = scale ] || [ "$KIND" = sweep ]; then
        aws s3 sync runs/world_model "$S3_ROOT/runs/world_model" "${SCALE_SYNC[@]}" --only-show-errors
    elif [ -d "$RUN_DIR" ]; then
        aws s3 sync "$RUN_DIR" "$RUN_URI" --only-show-errors
    fi
    aws s3 cp "$LOG" "$RUN_URI/job.log" --only-show-errors
}
finish() {
    local status=$?
    echo "job exited with status $status at $(date -u)"
    kill "$UPLOADER" 2>/dev/null
    echo "$status" > /tmp/job_status
    upload
    aws s3 cp /tmp/job_status "$RUN_URI/job_status" --only-show-errors
    shutdown -h now
}
UPLOADER=
trap finish EXIT

echo "job $KIND $NAME started at $(date -u) on $(curl -s http://169.254.169.254/latest/meta-data/instance-type 2>/dev/null)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"
curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
export PATH=$HOME/.local/bin:$PATH
uv sync -q || exit 1
aws s3 sync "$S3_ROOT/data" data --only-show-errors || exit 1

if [ "$KIND" = world_model ] || [ "$KIND" = policy ] || [ "$KIND" = scale ] || [ "$KIND" = sweep ]; then
    WANDB_API_KEY=$(aws secretsmanager get-secret-value --secret-id double_pendulum/wandb_api_key \
        --query SecretString --output text) || exit 1
    export WANDB_API_KEY
fi

(while sleep 300; do upload; done) &
UPLOADER=$!

case $KIND in
    bench)
        time uv run python -m world_model.train world_model/configs/base.toml --name bench-wm \
            --steps 5000 --wandb disabled || exit 1
        aws s3 sync "$S3_ROOT/runs/world_model/wm3" runs/world_model/wm3 --only-show-errors
        time uv run python -m imagination.train imagination/configs/base.toml --name bench-policy \
            --iterations 100 --set eval_every=50 --wandb disabled || exit 1
        uv run python -c "
import torch
t = torch.load('runs/policy/bench-policy/policy.pt', weights_only=False)['time']
print('policy seconds per phase over 100 iterations:', {k: round(v, 1) for k, v in t.items()})"
        ;;
    world_model)
        if aws s3 ls "$RUN_URI/" > /dev/null 2>&1; then
            echo "replacing what an interrupted run left at $RUN_URI"
            aws s3 rm "$RUN_URI" --recursive --only-show-errors
        fi
        uv run python -m world_model.train "$CONFIG" --name "$NAME" "${ARGS[@]}"
        ;;
    policy)
        WM=$(uv run python -c "import tomllib; print(tomllib.load(open('$CONFIG', 'rb'))['world_model'])")
        aws s3 sync "$S3_ROOT/$(dirname "$WM")" "$(dirname "$WM")" --only-show-errors || exit 1
        [ -f "$WM" ] || { echo "no world model at $S3_ROOT/$WM"; exit 1; }
        aws s3 sync "$RUN_URI" "$RUN_DIR" --only-show-errors
        LATEST=$(ls "$RUN_DIR"/checkpoints/iter_*.pt 2>/dev/null | tail -1)
        if [ -n "$LATEST" ]; then
            echo "resuming from $LATEST"
            uv run python -m imagination.train --resume "$LATEST"
        else
            rm -rf "$RUN_DIR"
            uv run python -m imagination.train "$CONFIG" --name "$NAME" "${ARGS[@]}"
        fi
        ;;
    run)
        uv run python -m "$CONFIG" "${ARGS[@]}"
        ;;
    sweep)
        mkdir -p runs/world_model
        aws s3 sync "$S3_ROOT/runs/world_model" runs/world_model "${SCALE_SYNC[@]}" --only-show-errors
        uv run python -m world_model.sweep "$CONFIG" "${ARGS[@]}"
        ;;
    scale)
        mkdir -p runs/world_model
        aws s3 sync "$S3_ROOT/runs/world_model" runs/world_model "${SCALE_SYNC[@]}" --only-show-errors
        uv run python -m world_model.scale "$CONFIG" --name "$NAME" "${ARGS[@]}"
        ;;
    *)
        echo "unknown kind $KIND"
        exit 1
        ;;
esac
