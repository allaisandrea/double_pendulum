"""Helpers the training programs share."""
import subprocess
import time
from contextlib import contextmanager

import torch


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
