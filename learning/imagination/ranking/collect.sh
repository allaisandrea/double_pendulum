#!/bin/bash
# Runs the ranking pool on the rig, from the repository root:
#
#   learning/imagination/ranking/collect.sh [SECONDS] [SEED] [POOL] [ACTIVE] [REST]
#
# Each policy in POOL (learning/imagination/ranking/pool.txt; the first
# word of each line is a policy's name), exported to
# learning/runs/ranking/<name>.json by `imagination.ranking export`, runs
# greedily for SECONDS (120), driving ACTIVE s (10) and resting REST s (5), without the
# speed governor, in an order shuffled with SEED (0). Each run starts after
# 15 s of rest, so from the pendulum hanging, and 120 s ends in a rest.
# recordings/ranking-<stamp>-seed<SEED>.tsv lists each policy and its
# recording, for `imagination.ranking imagine`; each run's log goes beside
# it. A failed run (the camera dropping, say) ends the session.
#
# `kill -INT $(cat recordings/ranking.pid)` stops the run under way as
# Ctrl-C would, motor first, and skips the rest.
cd "$(dirname "$0")/../../.." || exit 1
SECONDS_PER=${1:-120}
SEED=${2:-0}
POOL=${3:-learning/imagination/ranking/pool.txt}
ACTIVE=${4:-10}
REST=${5:-5}
STAMP=$(date +%s)
MANIFEST=recordings/ranking-$STAMP-seed$SEED.tsv
LOGS=recordings/ranking-$STAMP-seed$SEED-logs
mkdir -p "$LOGS"
echo $$ > recordings/ranking.pid
child=
trap '[ -n "$child" ] && kill -INT "$child" 2>/dev/null && wait "$child"; echo "stopped"; exit 130' INT TERM

cargo build -q --manifest-path harness/Cargo.toml || exit 1
ORDER=$(grep -v "^#" "$POOL" | awk 'NF {print $1}' | python3 -c "
import random, sys
names = sys.stdin.read().split()
random.Random($SEED).shuffle(names)
print(' '.join(names))")
for name in $ORDER; do
    [ -f "learning/runs/ranking/$name.json" ] || { echo "no export for $name: run imagination.ranking export"; exit 1; }
done
echo "order: $ORDER"
echo "manifest: $MANIFEST"

for name in $ORDER; do
    sleep 15 & child=$!
    wait "$child"
    echo "$(date +%H:%M:%S) $name"
    ./harness/target/debug/collect --policy "learning/runs/ranking/$name.json" --max-speed 0 \
        --active-s "$ACTIVE" --rest-s "$REST" --duration "$SECONDS_PER" > "$LOGS/$name.log" 2>&1 &
    child=$!
    wait "$child"
    status=$?
    child=
    recording=$(sed -n 's/^recording to \([^ ]*\) .*/\1/p' "$LOGS/$name.log")
    if [ $status -ne 0 ] || [ -z "$recording" ]; then
        echo "$name failed (status $status); see $LOGS/$name.log"
        exit 1
    fi
    printf '%s\t%s\n' "$name" "$PWD/$recording" >> "$MANIFEST"
done
echo "done: $MANIFEST"
