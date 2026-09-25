"""The trapezoidal learning-rate schedule of Hägele et al. 2024, "Scaling Laws
and Compute-Optimal Training Beyond Fixed Training Durations"
(arXiv:2405.18392): a linear warmup, a constant rate for most of training,
and a cooldown to zero shaped as 1 - sqrt, which they found beats a
linear one.
"""
import math

import torch


def trapezoid(step: int, steps: int, warmup: int, cooldown: int) -> float:
    """The learning-rate factor for optimizer step `step`, counted from 0 of
    `steps`: rising linearly over the first `warmup`, then 1, then falling as
    1 - sqrt over the last `cooldown`."""
    if step < warmup:
        return (step + 1) / warmup
    start = steps - cooldown
    if step >= start:
        return 1 - math.sqrt((step - start) / cooldown)
    return 1.0


def trapezoid_scheduler(opt, steps: int, warmup: int, cooldown_fraction: float):
    """A scheduler to step once after each of `steps` optimizer steps."""
    cooldown = round(steps * cooldown_fraction)
    if warmup + cooldown > steps:
        raise ValueError(f"warmup {warmup} and cooldown {cooldown} exceed {steps} steps")
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda step: trapezoid(step, steps, warmup, cooldown)
    )
