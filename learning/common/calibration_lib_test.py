import math

import numpy as np
import torch

from common.calibration_lib import DTYPE, Calibration, Poses, angles, fit, rotvec_to_matrix


def rig() -> Calibration:
    """A rig like the real one: the camera upside down 0.59 m from the axis."""
    c = Calibration()
    with torch.no_grad():
        c.camera_rotvec.copy_(torch.tensor([0.04, -0.04, math.pi - 0.03], dtype=DTYPE))
        c.camera_t.copy_(torch.tensor([-0.011, -0.002, 0.59], dtype=DTYPE))
        c.pivots.copy_(torch.tensor([[0.0, 0.089], [-0.0026, 0.112]], dtype=DTYPE))
        c.mount_t.copy_(torch.tensor([[-0.0005, 0.030, 0.0], [0.0, 0.021, -0.013], [-0.0002, 0.023, -0.024]], dtype=DTYPE))
        c.mount_rotvec.copy_(torch.tensor([[0.02, -0.01, -0.01], [-0.1, 0.02, 0.025], [0.0, 0.02, 0.02]], dtype=DTYPE))
    return c


def recording(truth: Calibration, name: str, seed: int) -> tuple[Poses, torch.Tensor]:
    """Drives and rests: the angles wander while driven and hang at 0 for
    the last 600 frames of each 1000, with pose noise as the real tags'."""
    g = torch.Generator().manual_seed(seed)
    t = torch.arange(6000, dtype=DTYPE)
    phi = torch.stack([0.8 * torch.sin(t / 37 + k) + 1.5 * torch.sin(t / (90 + 20 * k)) for k in range(3)], -1)
    driving = (t % 1000) < 400
    phi = torch.where(driving[:, None], phi, 0.0)
    action = np.where(driving.numpy(), 50, 0).astype(np.float32)
    with torch.no_grad():
        pos, rot = truth.predict(phi)
    pos = pos + torch.randn(pos.shape, generator=g, dtype=DTYPE) * torch.tensor([0.3e-3, 0.3e-3, 2e-3], dtype=DTYPE)
    noise = torch.randn(*rot.shape[:2], 3, generator=g, dtype=DTYPE) * torch.tensor([0.1, 0.1, 0.01], dtype=DTYPE)
    rot = rot @ rotvec_to_matrix(noise)
    present = torch.rand(pos.shape[:2], generator=g) > 0.03
    return Poses(name, pos, rot, present, action), phi


def test_the_fit_recovers_the_angles_and_a_new_session_needs_only_its_camera():
    torch.manual_seed(0)
    truth = rig()
    poses, phis = zip(*(recording(truth, f"r{k}", k) for k in range(3)))
    cal, cams = fit(list(poses), frames=3000, iterations=600, log=None)
    phi, _ = angles(cams["r1"], poses[1])
    seen = poses[1].present.all(-1)
    err = (phi - phis[1])[seen].abs()
    assert err.median() < math.radians(0.3) and err.quantile(0.99) < math.radians(1.5)

    # The camera moves 3 mm and turns 2 degrees: refitting the camera alone
    # brings the angles back.
    moved = rig()
    with torch.no_grad():
        moved.camera_t += torch.tensor([0.003, 0.0, 0.0], dtype=DTYPE)
        moved.camera_rotvec += torch.tensor([0.035, 0.0, 0.0], dtype=DTYPE)
    p, true_phi = recording(moved, "later", 7)
    cam, _ = fit([p], frames=2000, iterations=400, geometry=cal, log=None)
    phi, _ = angles(cam, p)
    seen = p.present.all(-1)
    assert (phi - true_phi)[seen].abs().median() < math.radians(0.3)
