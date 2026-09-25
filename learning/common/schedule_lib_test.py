import pytest
import torch

from common.schedule_lib import trapezoid, trapezoid_scheduler


def test_the_trapezoid_warms_up_holds_and_cools_down_to_near_zero():
    f = [trapezoid(i, 100, warmup=10, cooldown=20) for i in range(100)]
    assert f[0] == pytest.approx(0.1)
    assert f[9] == 1.0
    assert f[10:81] == [1.0] * 71
    assert f[90] == pytest.approx(1 - 0.5**0.5)
    assert 0 < f[99] < 0.25
    assert all(a >= b for a, b in zip(f[80:], f[81:]))


def test_the_scheduler_steps_the_optimizer_through_the_trapezoid():
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=2.0)
    sched = trapezoid_scheduler(opt, 100, warmup=10, cooldown_fraction=0.2)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] == pytest.approx(0.2)
    assert lrs[50] == pytest.approx(2.0)
    assert lrs[80] == pytest.approx(2.0)
    assert lrs[81] < 2.0
