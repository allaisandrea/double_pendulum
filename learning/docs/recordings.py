"""Writes docs/recordings.md: a table of every recording uploaded to S3, what
it holds, and where it is used.

    uv run python -m docs.recordings

The recordings are those under s3://allais-andrea-store/double_pendulum/data/,
read from their copies in data/ (aws s3 sync first). Each row comes from
the recording's own metadata and frames; its role from the ranking
manifests in ../recordings and from the configs that name it.
"""
import json
import re
import subprocess
import tomllib
from datetime import datetime
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.compute as pc

S3 = "s3://allais-andrea-store/double_pendulum/data/"
PROFILE = "andrea-personal"
DATA = Path("data")
MANIFESTS = Path("../recordings")
OUT = Path("docs/recordings.md")
# The first recording after a tag on the rig moved (imagination.loop.FIRST).
FIRST = 1790789841
CONFIGS = [*sorted(Path("world_model/configs").glob("*.toml")), *sorted(Path("imagination/configs").glob("*.toml"))]


def uploaded() -> list[str]:
    out = subprocess.run(["aws", "--profile", PROFILE, "s3", "ls", "--recursive", S3],
                         capture_output=True, text=True, check=True).stdout
    return sorted({m.group(1) for m in re.finditer(r"data/(\d+)/frames\.arrows", out)})


def greedy_tests() -> dict[str, str]:
    """Each recording a ranking manifest lists, and the manifest's name."""
    tests = {}
    for m in sorted(MANIFESTS.glob("ranking-*.tsv")):
        for line in m.read_text().splitlines():
            tests[Path(line.split("\t")[1]).name] = m.stem
    return tests


def uses() -> dict[str, dict[str, list[str]]]:
    """For each recording, by role, the configs and the cloud runs that name
    it: configs as wm:<name> and policy:<name>; runs, which name their data
    with --set (the loops' do), by their run names."""
    out: dict[str, dict[str, list[str]]] = {}

    def add(recs, role, name):
        for r in recs:
            out.setdefault(r, {}).setdefault(role, []).append(name)

    for path in CONFIGS:
        cfg = tomllib.loads(path.read_text())
        name = ("wm:" if path.parent.parent.name == "world_model" else "policy:") + path.stem
        add(cfg.get("train", []), "train", name)
        add(cfg.get("recordings", []), "starts", name)
        val = cfg.get("val", {})
        for role, recs in (val if isinstance(val, dict) else {"val": val}).items():
            add(recs, role, name)
    for job in sorted(Path("runs").glob("*/*/job.json")):
        for arg in json.loads(job.read_text())["job"]:
            key, _, value = arg.partition("=")
            if key in ("train", "recordings") and value.startswith("["):
                add(tomllib.loads(f"x = {value}")["x"], "train" if key == "train" else "starts", job.parent.name)
    return out


def driver(meta: dict) -> tuple[str, str, str]:
    """The policy, how it acted, and its actions."""
    text = meta["policy"]
    if text.startswith("random walk"):
        return "random walk", "random walk", f"±{meta['policy_range']}, steps ≤ {meta['policy_step']}"
    policy = Path(meta.get("policy_file", "?")).stem
    if policy == "policy":  # an export beside its checkpoint: runs/policy/<name>/policy.json
        policy = Path(meta["policy_file"]).parent.name
    levels = [int(a) for a in re.search(r"actions \[([^\]]*)\]", text).group(1).split(",")]
    mode = "sampled" if meta.get("policy_sample") == "true" else "greedy"
    burst = re.search(r"offset uniform in ±(\d+), clipped to ±(\d+)", text)
    if burst and float(meta.get("perturb_rate", 0)) > 0:
        mode += f", bursts ±{burst.group(1)} to ±{burst.group(2)}"
    return policy, mode, f"{len(levels)} to ±{max(levels)}"


def row(rec: str, tests: dict, used: dict) -> dict:
    with ipc.open_stream(DATA / rec / "frames.arrows") as reader:
        meta = {k.decode(): v.decode() for k, v in reader.schema.metadata.items()}
        table = reader.read_all()
    minutes = table["t_capture"][-1].value / 60e9
    seen = [1 - pc.sum(pc.is_null(table[f"tag{i}_pose"])).as_py() / len(table) for i in range(3)]
    policy, mode, actions = driver(meta)
    if rec in tests:
        role = f"greedy test ({tests[rec]})"
    elif policy == "random walk":
        role = "random walk"
    else:
        role = "collection"
    start = datetime.fromtimestamp(int(meta["t0_unix_ns"]) / 1e9)
    return {
        "recording": rec,
        "start": start.strftime("%m-%d %H:%M"),
        "min": f"{minutes:.0f}",
        "kind": role,
        "policy": policy,
        "actions": f"{mode}; {actions}",
        "drive/rest s": f"{int(meta['active_ns']) // 10**9}/{int(meta['rest_ns']) // 10**9}",
        "tags seen %": "/".join(f"{100 * s:.0f}" for s in seen),
        "used in": "; ".join(f"{role}: {', '.join(names)}" for role, names in used.get(rec, {}).items()),
        "_date": start.strftime("%Y-%m-%d"),
    }


def table(rows: list[dict]) -> list[str]:
    cols = [c for c in rows[0] if not c.startswith("_")]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(f"`{r[c]}`" if c == "recording" else r[c] for c in cols) + " |" for r in rows]
    return lines


def main():
    tests, used = greedy_tests(), uses()
    rows = [row(rec, tests, used) for rec in uploaded()]
    eras = [("The rig as it is now", [r for r in rows if int(r["recording"]) >= FIRST]),
            ("Before the tag moved (2026-09-30)", [r for r in rows if int(r["recording"]) < FIRST])]
    hours = lambda rs: sum(float(r["min"]) for r in rs) / 60
    lines = [
        "# Recordings",
        "",
        "Every recording uploaded to `s3://allais-andrea-store/double_pendulum/data/`, newest first,",
        "written by `uv run python -m docs.recordings` from the recordings themselves. On",
        "2026-09-30 a tag on the rig moved: recordings from `1790789841` on are of the rig as",
        "it is now, the earlier ones of the rig as it was, and the two are not mixed.",
        "",
        "- **kind**: a random walk (the stand-in policy), a collection with a trained policy",
        "  (training data), or a greedy test (named by its ranking manifest in `recordings/`).",
        "- **actions**: how the policy acted (greedy, sampled, with exploration bursts and",
        "  their offset and clip), then its number of actions and their range.",
        "- **drive/rest s**: the duty cycle, seconds driving then resting.",
        "- **tags seen %**: the share of frames each of tags 0, 1 and 2 was seen in.",
        "- **used in**: by role (`train`, a validation set, or `starts`, a policy's",
        "  episode starts), the configs that name the recording (`wm:` world model,",
        "  `policy:` policy) and the cloud runs that name it with `--set`, as the loops'",
        "  runs do (those in `runs/` here).",
    ]
    for title, rs in eras:
        rs = sorted(rs, key=lambda r: r["recording"], reverse=True)
        lines += ["", f"## {title}", "", f"{len(rs)} recordings, {hours(rs):.1f} hours."]
        for date in sorted({r["_date"] for r in rs}, reverse=True):
            day = [r for r in rs if r["_date"] == date]
            lines += ["", f"### {date}", "", f"{len(day)} recordings, {hours(day):.1f} hours.", "", *table(day)]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"{OUT}: {len(rows)} recordings, {hours(rows):.1f} hours")


if __name__ == "__main__":
    main()
