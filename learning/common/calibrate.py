"""Calibrates the recordings: from tag poses to link angles.

    uv run python -m common.calibrate geometry 2026-09-30
    uv run python -m common.calibrate angles                  # every recording in data/
    uv run python -m common.calibrate angles 1790897267 --force --upload
    uv run python -m common.calibrate report

The rig's eras are in calibration/eras.toml: a recording belongs to the
last era whose `first` it is not before. `geometry ERA` fits the era's
geometry (common.calibration_lib.fit) to --recordings of its recordings in
data/, spread evenly in time, and writes calibration/<ERA>.json.

`angles` fits each recording's camera, with its era's geometry held,
computes its link angles and writes them as data/<recording>/angles.arrows
(common.data_lib.write_angles), where world models and policies with
`observation = "angles"` read them. A recording whose angles file was made
from the same geometry is skipped, unless --force; --upload copies each new
file to the bucket. The file's metadata records the era, the geometry's
hash, the camera, and the residuals, which `report` lists: a tag that
slipped on its link shows as residuals far above the rest of its era's.
"""
import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import torch

from common.calibration_lib import NUM_TAGS, Calibration, angles, fit, load_poses
from common.data_lib import ANGLES_FILE, angles_metadata, write_angles

ERAS = Path("calibration/eras.toml")
DATA = Path("data")
S3 = "s3://allais-andrea-store/double_pendulum/data"
PROFILE = "andrea-personal"
CHANNELS = ["across", "along", "depth", "tilt_x", "tilt_y", "yaw"]


def eras() -> list[dict]:
    return sorted(tomllib.loads(ERAS.read_text())["era"], key=lambda e: e["first"])


def era_of(recording: str) -> dict:
    return [e for e in eras() if int(recording) >= e["first"]][-1]


def geometry_path(era: dict) -> Path:
    return ERAS.parent / f"{era['name']}.json"


def recordings() -> list[str]:
    return sorted(p.parent.name for p in DATA.glob("*/frames.arrows"))


def fit_geometry(args):
    era = [e for e in eras() if e["name"] == args.era][0]
    names = [r for r in recordings() if era_of(r)["name"] == era["name"]]
    # Evenly in time, the longest of each stretch: they hold the most rests.
    picks = [max(chunk, key=lambda r: (DATA / r / "frames.arrows").stat().st_size)
             for chunk in np.array_split(np.array(names), min(args.recordings, len(names)))]
    print(f"{era['name']}: fitting the geometry to {len(picks)} of its {len(names)} recordings: {', '.join(picks)}")
    cal, _ = fit([load_poses(DATA / r / "frames.arrows") for r in picks], frames=args.frames)
    cal.save(geometry_path(era))
    k, m = cal.pivots.detach() * 1000, cal.mount_t.detach() * 1000
    print(f"pivots (mm): {k[0, 1]:.2f} along link 0; {k[1].numpy().round(2)} on link 1")
    for i in range(NUM_TAGS):
        print(f"tag {i}: at {m[i].numpy().round(2)} mm on its link, turned "
              f"{np.degrees(cal.mount_rotvec[i].detach().numpy()).round(2)} deg")
    print(f"wrote {geometry_path(era)}")


def geometry_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def calibrate_angles(args):
    names = args.recordings or recordings()
    for name in names:
        era = era_of(name)
        geometry = geometry_path(era)
        if not geometry.exists():
            raise SystemExit(f"{geometry} missing: run `common.calibrate geometry {era['name']}` first")
        out = DATA / name / ANGLES_FILE
        digest = geometry_hash(geometry)
        if out.exists() and not args.force and angles_metadata(out).get("geometry_hash") == digest:
            continue
        poses = load_poses(DATA / name / "frames.arrows")
        camera, _ = fit([poses], frames=2000, iterations=600, geometry=Calibration.load(geometry), log=None)
        phi, r = angles(camera, poses)
        phi = torch.where(poses.present, phi, float("nan")).numpy()
        seen = poses.present.numpy()
        med = np.stack([np.median(np.abs(r[seen[:, i], i].numpy()), 0) for i in range(NUM_TAGS)])
        wild = np.array([(np.abs(r[seen[:, i], i].numpy()) > 6).any(-1).mean() for i in range(NUM_TAGS)])
        meta = {
            "era": era["name"],
            "geometry_hash": digest,
            "camera": json.dumps({k: v for k, v in camera.to_json().items() if k.startswith("camera")}),
            "median_residual": json.dumps({CHANNELS[c]: med[:, c].round(3).tolist() for c in range(len(CHANNELS))}),
            "wild_share": json.dumps(wild.round(4).tolist()),
        }
        write_angles(out, phi, meta)
        print(f"{name} ({era['name']}): median |residual| across/yaw per tag "
              f"{med[:, [0, 5]].round(2).tolist()}, over 6 sigma {wild.round(4).tolist()}", flush=True)
        if args.upload:
            subprocess.run(["aws", "--profile", PROFILE, "s3", "cp", str(out), f"{S3}/{name}/{ANGLES_FILE}",
                            "--only-show-errors"], check=True)


def report(args):
    print("recording   era          median |residual| (sigmas) across / yaw per tag      over 6 sigma")
    for name in recordings():
        path = DATA / name / ANGLES_FILE
        if not path.exists():
            print(f"{name}  (no angles)")
            continue
        m = angles_metadata(path)
        med = json.loads(m["median_residual"])
        stale = "" if m["geometry_hash"] == geometry_hash(geometry_path(era_of(name))) else "  (stale geometry)"
        print(f"{name}  {m['era']:11s}  across {med['across']}  yaw {med['yaw']}  {json.loads(m['wild_share'])}{stale}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    g = sub.add_parser("geometry", help="fit an era's geometry")
    g.add_argument("era")
    g.add_argument("--recordings", type=int, default=8)
    g.add_argument("--frames", type=int, default=8000)
    a = sub.add_parser("angles", help="write recordings' calibrated angles")
    a.add_argument("recordings", nargs="*")
    a.add_argument("--force", action="store_true")
    a.add_argument("--upload", action="store_true")
    sub.add_parser("report", help="list the recordings' calibration residuals")
    args = parser.parse_args()
    {"geometry": fit_geometry, "angles": calibrate_angles, "report": report}[args.command](args)


if __name__ == "__main__":
    main()
