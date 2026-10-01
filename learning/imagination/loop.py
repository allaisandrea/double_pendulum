"""The world model / policy / data loop, the rig collecting while AWS trains.

    caffeinate -is uv run python -m imagination.loop               # both workers
    caffeinate -is uv run python -m imagination.loop --rig --chunks 8 --chunk-s 1800
    caffeinate -is uv run python -m imagination.loop --train --max-it 4

The data are every recording of the rig as it is now: those under
../recordings from FIRST on, sampled collections and greedy tests alike.

The rig worker collects without pause, in --chunk-s recordings driving
--active s and resting --rest s, sampling the latest policy. When a new
policy is ready, it finishes the recording under way, tests the policy
greedily for --greedy-s, and then collects with it. With --chunks it stops
after that many collections.

The training worker runs iterations: rb2-it{i}, a stochastic world model
(world_model/configs/rebootstrap.toml, --wm-steps) on all the data so
far; then ppo-rb2-it{i}, a 512 x 3 annealed policy
(imagination/configs/rebootstrap.toml, 5k iterations) in it, its episodes
starting from the same data; then the export and its check against
PyTorch. The next iteration starts as soon as the rig has tested the
policy, or at once if the rig worker is not running.

A failed collection (a camera drop) stops the rig worker only. Progress is
in runs/loop/state.json, the log in runs/loop/loop.log; running again
resumes from it, and a cloud job still running is waited on, not started
again. `kill -INT $(cat runs/loop/pid)` stops everything, the motor first.
"""
import argparse
import json
import os
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path

L = Path(__file__).resolve().parent.parent
REPO = L.parent
OUT = L / "runs" / "loop"
STATE, LOG, PID = OUT / "state.json", OUT / "loop.log", OUT / "pid"
PROFILE = "andrea-personal"
S3 = "s3://allais-andrea-store/double_pendulum"
# The code cloud jobs run: the commit with the rebootstrap configs.
CODE = "35ee37385b10df4988bb0b8c84390924ac8ab5b6"
INSTANCES = "g6.xlarge,g5.xlarge,g6e.xlarge,g4dn.xlarge"
# The first recording after a tag on the rig moved (2026-09-30).
FIRST = 1790789841
POLICY_ARGS = ["--set", "hidden=512", "--set", "layers=3", "--set", "iterations=5000"]

lock = threading.RLock()
children: set[subprocess.Popen] = set()
stopping = threading.Event()
training_done, rig_stopped = threading.Event(), threading.Event()


class Stop(Exception):
    pass


def log(msg: str):
    line = f"{time.strftime('%m-%d %H:%M:%S')} [{threading.current_thread().name}] {msg}"
    with lock:
        print(line, flush=True)
        with open(LOG, "a") as f:
            f.write(line + "\n")


def run(cmd, cwd=L, out=None, check=True) -> str:
    """Runs cmd, its output to `out` (a path) or captured; returns the output."""
    if stopping.is_set():
        raise Stop("stopping")
    log("$ " + " ".join(map(str, cmd))[:300])
    f = open(out, "w") if out else subprocess.PIPE
    child = subprocess.Popen(list(map(str, cmd)), cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
    with lock:
        children.add(child)
    text = child.communicate()[0] if not out else None
    code = child.wait()
    with lock:
        children.discard(child)
    if out:
        f.close()
        text = Path(out).read_text()
    if check and code != 0:
        raise Stop(f"{Path(str(cmd[0])).name} exited with {code}: {text[-800:]}")
    return text


def recordings() -> list[str]:
    """Every finished recording of the rig as it is now: those staged in
    learning/data, which a recording reaches once its run has ended."""
    return sorted(d.name for d in (REPO / "recordings").iterdir()
                  if d.name.isdigit() and int(d.name) >= FIRST and (L / "data" / d.name / "frames.arrows").exists())


# The state: the policies ready and tested, the policy the rig drives, the
# finished steps of each iteration, the data each iteration trained on, and
# the next collection's seed.
def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"ready": [], "tested": [], "rig_policy": None, "done": [], "data": {}, "seed": 400}


state: dict = {}


def save():
    with lock:
        STATE.write_text(json.dumps(state, indent=1))


def stage(rec: str):
    """Copies a recording into learning/data and to S3."""
    src = REPO / "recordings" / rec / "frames.arrows"
    dst = L / "data" / rec / "frames.arrows"
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    uri = f"{S3}/data/{rec}/frames.arrows"
    if subprocess.run(["aws", "--profile", PROFILE, "s3", "ls", uri], capture_output=True).returncode:
        run(["aws", "--profile", PROFILE, "s3", "cp", dst, uri, "--only-show-errors"])


def toml_list(xs):
    return "[" + ", ".join(f'"{x}"' for x in xs) + "]"


# The rig worker.
def score(manifest: str, name: str):
    """Logs a greedy test's score, against the hanging pose of the data."""
    text = run(["uv", "run", "-q", "python", "-c", f"""
import numpy as np
from pathlib import Path
from common.data_lib import hanging_yaws, load_recordings
from imagination.ranking_lib import episodes
h = hanging_yaws(load_recordings(Path('data'), {recordings()!r})).numpy()
e = [x.real for line in (Path('..') / {manifest!r}).read_text().splitlines()
     for x in episodes(Path(line.split(chr(9))[1]) / 'frames.arrows', h, 16)]
print(f'{name} greedy on the rig: {{np.mean(e):+.3f}} +- {{np.std(e, ddof=1) / len(e) ** 0.5:.3f}} over {{len(e)}} drives')
"""], check=False).strip()
    log("RESULT " + (text.splitlines()[-1] if text else "no score"))


def greedy_test(args, name: str):
    pool = OUT / f"pool-{name}.txt"
    pool.write_text(name + "\n")
    time.sleep(15)
    text = run([L / "imagination/ranking/collect.sh", args.greedy_s, 0, pool, args.active, args.rest], cwd=REPO,
               out=OUT / f"greedy-{name}.log")
    manifest = [l.split()[1] for l in text.splitlines() if l.startswith("done: ")][0]
    stage(Path((REPO / manifest).read_text().splitlines()[0].split("\t")[1]).name)
    with lock:
        state["tested"].append(name)
        state["rig_policy"] = name
        save()
    score(manifest, name)


def collect_one(args):
    with lock:
        policy, seed = state["rig_policy"], state["seed"]
        state["seed"] += 1
        save()
    time.sleep(15)  # the pendulum hangs still before each run
    text = run([REPO / "harness/target/debug/collect", "--policy", L / f"runs/ranking/{policy}.json", "--policy-sample",
                "--max-speed", "0", "--active-s", args.active, "--rest-s", args.rest, "--duration", args.chunk_s,
                "--seed", seed], cwd=REPO, out=OUT / f"collect-{seed}.log")
    rec = Path([l for l in text.splitlines() if l.startswith("recording to ")][0].split()[2]).name
    if stopping.is_set():
        raise Stop(f"{rec} was cut short by the stop: not staged")
    still = motionless_minutes(rec)
    if still:
        raise Stop(f"{rec}: the motor arm hardly moved while driven in minutes {still} (motor supply?): not staged")
    stage(rec)
    tags = "; ".join(l.strip() for l in text.splitlines() if "tag " in l and "frames (" in l)
    log(f"recorded {rec} with {policy}: {tags}")


def motionless_minutes(rec: str, min_rev_s: float = 0.2) -> list[int]:
    """The minutes of a recording in which the motor arm's median speed,
    over the frames the policy drove, stayed under `min_rev_s`: a policy
    drives it at about 0.5 rev/s, and an unpowered motor leaves it at the
    0.07 of sensor noise."""
    text = run(["uv", "run", "-q", "python", "-c", f"""
import numpy as np
from pathlib import Path
from common.data_lib import load_recording
r = load_recording(Path('../recordings/{rec}/frames.arrows'))
ang = np.arctan2(r.obs[:, 0, 0], r.obs[:, 0, 1])
speed = np.abs(np.angle(np.exp(1j * np.diff(ang)))) * 125 / (2 * np.pi)
ok = r.present[1:, 0] & r.present[:-1, 0] & (r.action[1:] != 0)
minute = np.arange(len(speed)) // (60 * 125)
print([int(m) for m in np.unique(minute) if (ok & (minute == m)).sum() > 500
       and np.median(speed[ok & (minute == m)]) < {min_rev_s}])
"""])
    return json.loads(text.strip().splitlines()[-1])


def rig_worker(args):
    chunks = 0
    try:
        while not stopping.is_set():
            untested = [p for p in state["ready"] if p not in state["tested"]]
            if untested:
                greedy_test(args, untested[-1])
                with lock:  # superseded before the rig reached them
                    state["tested"] += [p for p in untested[:-1] if p not in state["tested"]]
                    save()
            elif (args.with_training and training_done.is_set()) or (args.chunks and chunks >= args.chunks):
                break
            else:
                collect_one(args)
                chunks += 1
        log("rig worker finished")
    except Stop as e:
        log(f"RIG STOPPED: {e}")
    finally:
        rig_stopped.set()


# The training worker.
def cloud_job(line: list[str], kind_dir: str, name: str):
    """Runs one job through the queue, pinned to CODE, and checks it succeeded."""
    qfile = OUT / f"queue-{name}.txt"
    qfile.write_text(shlex.join(line) + "\n")
    run(["uv", "run", "python", "-u", L / "cloud/queue.py", qfile, "--spot", "2", "--on-demand", "2", "--code", CODE],
        out=OUT / f"queue-{name}.log")
    status = subprocess.run(["aws", "--profile", PROFILE, "s3", "cp", f"{S3}/runs/{kind_dir}/{name}/job_status", "-"],
                            capture_output=True, text=True).stdout.strip()
    if status != "0":
        raise Stop(f"{name} ended with status {status!r}")
    run(["aws", "--profile", PROFILE, "s3", "sync", f"{S3}/runs/{kind_dir}/{name}", L / "runs" / kind_dir / name,
         "--only-show-errors", "--exclude", "*.mp4"])


def step(key, fn):
    if key in state["done"]:
        return
    log(f"== {key}")
    fn()
    with lock:
        state["done"].append(key)
        save()


def iteration(args, i: int):
    with lock:
        data = state["data"].setdefault(str(i), recordings())
        save()
    for rec in data:
        stage(rec)
    wm, policy = f"rb2-it{i}", f"ppo-rb2-it{i}"
    step(f"wm{i}", lambda: (log(f"{wm} on {len(data)} recordings"), cloud_job(
        ["world_model", wm, "world_model/configs/rebootstrap.toml", "--instance", INSTANCES, "--",
         "--steps", args.wm_steps, "--set", f"train={toml_list(data)}"], "world_model", wm)))
    step(f"policy{i}", lambda: cloud_job(
        ["policy", policy, "imagination/configs/rebootstrap.toml", "--instance", INSTANCES, "--", *POLICY_ARGS,
         "--set", f'world_model="runs/world_model/{wm}/model.pt"', "--set", f"recordings={toml_list(data)}"],
        "policy", policy))

    def check():
        run(["uv", "run", "-q", "python", "-m", "imagination.export", f"runs/policy/{policy}/policy.pt",
             "--out", f"runs/ranking/{policy}.json"])
        out = run([REPO / "harness/target/debug/check_policy", L / f"runs/ranking/{policy}.json"], cwd=REPO)
        if "matches PyTorch" not in out:
            raise Stop(f"{policy}'s export does not match PyTorch: {out[-500:]}")
        last = [l for l in (L / f"runs/policy/{policy}/job.log").read_text().splitlines() if l.startswith("iteration")]
        log(f"{policy} ready; in imagination: {last[-1][:160] if last else '?'}")
        with lock:
            state["ready"].append(policy)
            save()

    step(f"check{i}", check)
    # The next iteration starts once the rig has tested this policy.
    while policy not in state["tested"] and not rig_stopped.is_set() and not stopping.is_set():
        time.sleep(10)


def training_worker(args):
    try:
        for i in range(1, args.max_it + 1):
            iteration(args, i)
        log("training worker finished")
    except Stop as e:
        log(f"TRAINING STOPPED: {e}")
    finally:
        training_done.set()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rig", action="store_true", help="run the rig worker (with --train or alone)")
    parser.add_argument("--train", action="store_true", help="run the training worker (with --rig or alone)")
    parser.add_argument("--chunk-s", type=int, default=1200, help="seconds per collection")
    parser.add_argument("--chunks", type=int, default=0, help="stop the rig after this many collections (0: never)")
    parser.add_argument("--greedy-s", type=int, default=300, help="seconds of each greedy test")
    parser.add_argument("--active", type=int, default=20, help="seconds driving in each cycle")
    parser.add_argument("--rest", type=int, default=8, help="seconds resting in each cycle")
    parser.add_argument("--max-it", type=int, default=6, help="the last iteration to train")
    parser.add_argument("--wm-steps", type=int, default=250000, help="world model training steps")
    args = parser.parse_args()
    both = not (args.rig or args.train)
    # The rig stops with the training only when both run.
    args.with_training = both

    OUT.mkdir(parents=True, exist_ok=True)
    state.update(load_state())
    PID.write_text(str(os.getpid()))

    def stop(signum, frame):
        stopping.set()
        with lock:
            for c in list(children):
                c.send_signal(signal.SIGINT)
        log("stopping by signal")

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    save()
    run(["cargo", "build", "-q", "--manifest-path", REPO / "harness/Cargo.toml"], cwd=REPO)
    threads = []
    if both or args.rig:
        if not state["rig_policy"]:
            raise SystemExit("no policy for the rig yet: set rig_policy in the state")
        threads.append(threading.Thread(target=rig_worker, args=(args,), name="rig"))
    else:
        rig_stopped.set()
    if both or args.train:
        threads.append(threading.Thread(target=training_worker, args=(args,), name="train"))
    else:
        training_done.set()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        time.sleep(1)
    log("loop finished")


if __name__ == "__main__":
    main()
