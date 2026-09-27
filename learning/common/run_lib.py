"""Helpers the training programs share."""
import subprocess
import time
import tomllib
from contextlib import contextmanager
from pathlib import Path

import torch


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: Path, overrides: list[str], optional=frozenset()) -> dict:
    """A TOML config, with each KEY=VALUE of `overrides` (--set) replacing
    a key it has, or adding one of `optional`; VALUE is TOML."""
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep or (key not in cfg and key not in optional):
            raise SystemExit(f"--set {item}: not KEY=VALUE for a key in the config")
        cfg[key] = tomllib.loads(f"v = {value}")["v"]
    return cfg


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The latest of a run's checkpoints, named with zero-padded steps or
    iterations, or None."""
    checkpoints = sorted((Path(run_dir) / "checkpoints").glob("*.pt"))
    return checkpoints[-1] if checkpoints else None


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "describe", "--always", "--dirty"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def synchronize(device: torch.device):
    """Waits for the work queued on `device`, so a timer reads real time."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


class Timer:
    """Adds up the seconds spent in each named phase, per iteration and
    since training began."""

    def __init__(self, device: torch.device, totals: dict | None = None):
        self.device = device
        self.totals = dict(totals or {})
        self.current: dict[str, float] = {}

    @contextmanager
    def __call__(self, phase: str):
        synchronize(self.device)
        started = time.perf_counter()
        try:
            yield
        finally:
            synchronize(self.device)
            seconds = time.perf_counter() - started
            self.current[phase] = self.current.get(phase, 0.0) + seconds
            self.totals[phase] = self.totals.get(phase, 0.0) + seconds

    def metrics(self) -> dict:
        """This iteration's seconds per phase and in all, and each phase's
        share of all the time timed so far; then starts a new iteration."""
        out = {f"time/{k}_s": v for k, v in self.current.items()}
        out["time/iteration_s"] = sum(self.current.values())
        total = sum(self.totals.values())
        out |= {f"time/share/{k}": v / total for k, v in self.totals.items()}
        out["time/total_s"] = total
        self.current = {}
        return out


def rng_state(device: torch.device) -> dict:
    """torch's global generators, on the CPU and on `device`."""
    state = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["cuda"] = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        state["mps"] = torch.mps.get_rng_state()
    return state


def set_rng_state(state: dict, device: torch.device):
    torch.set_rng_state(state["cpu"])
    if "cuda" in state and device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    elif "mps" in state and device.type == "mps":
        torch.mps.set_rng_state(state["mps"])
