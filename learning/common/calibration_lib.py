"""A kinematic calibration of the rig: from the tags' poses to the links' angles.

The pendulum is three rigid links turning in parallel planes: link 0 about
the motor axis, link 1 about a pivot on link 0, link 2 about a pivot on
link 1. In the rig's frame (origin on the motor axis in tag 0's plane, z along the
axis, y from it towards link 1's pivot), link i's angle phi_i is its turn
about z from hanging: 0 when the chain hangs still, the same for every
link, so the angles do not depend on where the tags sit or how the camera
looks.

    p_0 = 0,  p_1 = p_0 + Rz(phi_0) k_0,  p_2 = p_1 + Rz(phi_1) k_1
    tag i:  position  T (p_i + Rz(phi_i) m_i),  rotation  R Rz(phi_i) M_i

with T = (R, t) the camera's pose of the rig frame, k_i each pivot's
offset on its link (in the plane; k_0 along y, by the frame's definition),
and m_i, M_i where and how tag i is mounted on its link (m_0 in the
plane z = 0, by the same). A calibration is these 25 numbers; `fit` estimates
them from recorded poses, with every frame's angles, by least squares on
the tags' positions and rotations, each weighted by its noise (in-plane
position and yaw are precise; depth and tilt much less so) and made
robust to the detector's occasional flips. Gravity sets the angles' zero:
the mean angle of every link over the ends of the rests is 0.

`angles` then gives each frame's three angles from its poses by
Gauss-Newton, with the calibration fixed: every seen tag counts, through
both where it is and how it is turned.
"""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.ipc as ipc
import torch

NUM_TAGS = 3
# Noise of a pose at rest, in the camera's frame: in-plane position (m),
# depth (m), yaw (rad), tilt (rad).
SIGMA_PLANE, SIGMA_DEPTH = 0.5e-3, 3e-3
SIGMA_YAW, SIGMA_TILT = math.radians(0.7), math.radians(8.0)
# Residuals beyond this many sigmas count linearly (a Huber loss).
HUBER = 3.0
DTYPE = torch.float64


def rotvec_to_matrix(v: torch.Tensor) -> torch.Tensor:
    """Rodrigues: [..., 3] -> [..., 3, 3]."""
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    k = v / theta
    K = torch.zeros(*v.shape[:-1], 3, 3, dtype=v.dtype, device=v.device)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s, c = theta.sin()[..., None], theta.cos()[..., None]
    eye = torch.eye(3, dtype=v.dtype, device=v.device).expand_as(K)
    return eye + s * K + (1 - c) * K @ K


def matrix_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """The inverse of rotvec_to_matrix, for angles below pi."""
    cos = ((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    theta = cos.acos()
    w = torch.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    scale = torch.where(theta > 1e-6, theta / (2 * theta.sin().clamp(min=1e-12)), 0.5 + theta ** 2 / 12)
    return w * scale[..., None]


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """[..., 4] (w, x, y, z) -> [..., 3, 3]."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rz(phi: torch.Tensor) -> torch.Tensor:
    """Rotations about z [..., 3, 3] by phi [...]."""
    c, s = phi.cos(), phi.sin()
    z, o = torch.zeros_like(phi), torch.ones_like(phi)
    return torch.stack([torch.stack([c, -s, z], -1), torch.stack([s, c, z], -1), torch.stack([z, z, o], -1)], -2)


@dataclass
class Poses:
    """One recording's tag poses, one row per camera frame as data_lib lays
    them out: positions [T, NUM_TAGS, 3] (m), rotations [T, NUM_TAGS, 3, 3]
    (tag to camera), whether each tag was seen, and the action."""
    name: str
    position: torch.Tensor
    rotation: torch.Tensor
    present: torch.Tensor
    action: np.ndarray


def load_poses(path: Path) -> Poses:
    """A frames.arrows table's poses, with a row for every camera frame (a
    dropped one has no tag seen), as common.data_lib.load_recording."""
    with ipc.open_stream(path) as reader:
        table = reader.read_all()
    frame = table["frame"].to_numpy()
    slot = (frame - frame[0]).astype(np.int64)
    steps = int(slot[-1]) + 1
    pose = np.zeros((steps, NUM_TAGS, 7))
    present = np.zeros((steps, NUM_TAGS), bool)
    for tag in range(NUM_TAGS):
        col = table[f"tag{tag}_pose"].combine_chunks()
        values = col.values.slice(col.offset * 7, len(col) * 7).to_numpy(zero_copy_only=False).reshape(-1, 7)
        seen = col.is_valid().to_numpy(zero_copy_only=False)
        pose[slot[seen], tag] = values[seen]
        present[slot[seen], tag] = True
    pose[~present] = [0, 0, 1, 1, 0, 0, 0]
    action = np.zeros(steps, np.float32)
    action[slot] = table["action"].to_numpy()
    has_row = np.zeros(steps, bool)
    has_row[slot] = True
    action = action[np.maximum.accumulate(np.where(has_row, np.arange(steps), 0))]
    p = torch.from_numpy(pose).to(DTYPE)
    return Poses(path.parent.name, p[..., :3], quat_to_matrix(p[..., 3:]), torch.from_numpy(present), action)


def rest_ends(action: np.ndarray, min_rest: int = 500, last: int = 125) -> np.ndarray:
    """The frames in the last `last` of every rest of at least `min_rest`
    frames at action 0 that a later drive ends (not the recording's end)."""
    edges = np.flatnonzero(np.diff(np.r_[0, (action == 0).astype(int), 0]))
    out = [np.arange(e - last, e) for s, e in zip(edges[::2], edges[1::2]) if e - s >= min_rest and e < len(action)]
    return np.concatenate(out) if out else np.zeros(0, int)


class Calibration(torch.nn.Module):
    """The rig's 25 numbers (see the module docstring); `pivots[0, 0]` and
    `mount_t[0, 2]` are held at 0 by the frame's definition."""

    def __init__(self):
        super().__init__()
        self.camera_rotvec = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
        self.camera_t = torch.nn.Parameter(torch.zeros(3, dtype=DTYPE))
        self.pivots = torch.nn.Parameter(torch.zeros(2, 2, dtype=DTYPE))  # k_0, k_1 in the plane
        self.mount_t = torch.nn.Parameter(torch.zeros(NUM_TAGS, 3, dtype=DTYPE))
        self.mount_rotvec = torch.nn.Parameter(torch.zeros(NUM_TAGS, 3, dtype=DTYPE))

    def predict(self, phi: torch.Tensor, camera_rotvec=None, camera_t=None):
        """Each tag's position [B, NUM_TAGS, 3] and rotation [B, NUM_TAGS, 3,
        3] in the camera's frame, for link angles phi [B, NUM_TAGS]; the
        camera's pose is the calibration's, or camera_rotvec and camera_t,
        [3] or one per frame [B, 3]."""
        rv = self.camera_rotvec if camera_rotvec is None else camera_rotvec
        t = self.camera_t if camera_t is None else camera_t
        R = rotvec_to_matrix(rv)
        if R.dim() == 2:
            R = R.expand(len(phi), 3, 3)
        t = t.expand(len(phi), 3)
        Rl = rz(phi)  # [B, 3, 3, 3]
        # The frame's definition: k_0 along y, and tag 0 in the plane z = 0.
        gauge = torch.tensor([[0.0, 1.0], [1.0, 1.0]], dtype=DTYPE, device=phi.device)
        k = torch.cat([self.pivots * gauge, torch.zeros(2, 1, dtype=DTYPE, device=phi.device)], -1)
        mount_t = self.mount_t * torch.tensor([[1.0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=DTYPE, device=phi.device)
        p1 = Rl[:, 0] @ k[0]
        p2 = p1 + Rl[:, 1] @ k[1]
        pivots = torch.stack([torch.zeros_like(p1), p1, p2], 1)
        rig = pivots + (Rl @ mount_t[None, :, :, None]).squeeze(-1)
        position = (R[:, None] @ rig[..., None]).squeeze(-1) + t[:, None]
        rotation = R[:, None] @ Rl @ rotvec_to_matrix(self.mount_rotvec)
        return position, rotation

    def residuals(self, phi, position, rotation, present, camera_rotvec=None, camera_t=None):
        """Each seen tag's residuals [B, NUM_TAGS, 6], in sigmas: position
        across the image (2), depth, tilt (2) and yaw, all in the camera's
        frame but the rotation's, which is in the tag's."""
        pos, rot = self.predict(phi, camera_rotvec, camera_t)
        dp = position - pos
        dr = matrix_to_rotvec(rot.transpose(-1, -2) @ rotation)
        r = torch.cat([dp[..., :2] / SIGMA_PLANE, dp[..., 2:] / SIGMA_DEPTH, dr[..., :2] / SIGMA_TILT,
                       dr[..., 2:] / SIGMA_YAW], -1)
        return r * present[..., None]

    def to_json(self) -> dict:
        return {k: v.detach().cpu().tolist() for k, v in self.state_dict().items()}

    @classmethod
    def from_json(cls, d: dict) -> "Calibration":
        c = cls()
        c.load_state_dict({k: torch.tensor(v, dtype=DTYPE) for k, v in d.items()})
        return c

    def save(self, path: Path):
        Path(path).write_text(json.dumps(self.to_json(), indent=1))

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        return cls.from_json(json.loads(Path(path).read_text()))


def huber(r: torch.Tensor) -> torch.Tensor:
    a = r.abs()
    return torch.where(a < HUBER, 0.5 * a * a, HUBER * (a - 0.5 * HUBER))


def tag_yaw(rotation: torch.Tensor) -> torch.Tensor:
    """data_lib.yaw from rotation matrices."""
    return torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])


def initial(poses: list[Poses]) -> tuple[Calibration, list[torch.Tensor]]:
    """A starting calibration and starting angles: the plane from tag 0's
    positions, the axis at the centre of its circle, gravity from the
    hanging chain, and each frame's angles from the tags' yaws."""
    pos = torch.cat([p.position[p.present[:, 0], 0] for p in poses])
    centre = pos.mean(0)
    _, _, vt = torch.linalg.svd(pos - centre, full_matrices=False)
    normal = vt[2] * torch.sign(vt[2, 2])  # towards +z of the camera
    # The circle's centre in the plane (algebraic fit).
    e1 = torch.linalg.cross(normal, torch.tensor([1.0, 0, 0], dtype=DTYPE))
    e1 = e1 / e1.norm()
    e2 = torch.linalg.cross(normal, e1)
    uv = torch.stack([(pos - centre) @ e1, (pos - centre) @ e2], -1)
    A = torch.cat([2 * uv, torch.ones(len(uv), 1, dtype=DTYPE)], -1)
    sol = torch.linalg.lstsq(A, (uv ** 2).sum(-1, keepdim=True)).solution.squeeze(-1)
    axis = centre + sol[0] * e1 + sol[1] * e2
    # The frame's y: from the axis towards the hanging chain (tag 2).
    hang = [p.position[rest_ends(p.action)] for p in poses]
    hang_present = [p.present[rest_ends(p.action)] for p in poses]
    h = torch.cat(hang)
    hp = torch.cat(hang_present)
    tip = h[hp[:, 2], 2].mean(0)
    down = tip - axis
    down = down - (down @ normal) * normal
    y = down / down.norm()
    x = torch.linalg.cross(y, normal)
    R = torch.stack([x, y, normal], -1)  # rig axes in camera coordinates
    cal = Calibration()
    with torch.no_grad():
        cal.camera_rotvec.copy_(matrix_to_rotvec(R))
        cal.camera_t.copy_(axis)
        # Hanging, every link points down: its tag's position on it, and its
        # rotation, straight from the hanging poses.
        rig_h = (h - axis) @ R
        means = torch.stack([(rig_h[hp[:, i], i]).mean(0) for i in range(NUM_TAGS)])
        cal.pivots.copy_(torch.tensor([[0.0, means[1, 1] * 0.5 + means[0, 1] * 0.5],
                                       [0.0, means[2, 1] * 0.5 + means[1, 1] * 0.5]], dtype=DTYPE))
        cal.pivots[1, 1] -= cal.pivots[0, 1]
        cal.mount_t[0] = means[0]
        cal.mount_t[1] = means[1] - torch.cat([cal.pivots[0], torch.zeros(1, dtype=DTYPE)])
        cal.mount_t[2] = means[2] - torch.cat([cal.pivots[0] + cal.pivots[1], torch.zeros(1, dtype=DTYPE)])
        rot_h = torch.cat([p.rotation[rest_ends(p.action)] for p in poses])
        for i in range(NUM_TAGS):
            Mi = R.T @ rot_h[hp[:, i], i]
            cal.mount_rotvec[i] = matrix_to_rotvec(_mean_rotation(Mi))
    hang_yaw = torch.stack([_circ_mean(tag_yaw(rot_h[hp[:, i], i])) for i in range(NUM_TAGS)])
    phis = []
    for p in poses:
        yaw = tag_yaw(p.rotation)
        # The camera looks along +z of the rig or against it: yaw turns with
        # phi one way or the other.
        sign = torch.sign(R[2, 2])
        phi = _wrap(sign * (yaw - hang_yaw))
        phis.append(torch.where(p.present, phi, 0.0))
    return cal, phis


def _mean_rotation(Rs: torch.Tensor) -> torch.Tensor:
    u, _, vt = torch.linalg.svd(Rs.mean(0))
    d = torch.sign(torch.linalg.det(u @ vt))
    return u @ torch.diag(torch.tensor([1, 1, d], dtype=DTYPE)) @ vt


def _circ_mean(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a.sin().mean(), a.cos().mean())


def _wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(a.sin(), a.cos())


def fit(poses: list[Poses], frames: int = 6000, iterations: int = 2000, seed: int = 0, geometry: Calibration | None = None,
        log=print) -> tuple[Calibration, dict[str, Calibration]]:
    """Fits the rig's geometry, and a camera pose for each recording (the
    camera drifts by millimetres between sessions; the rig does not), to a
    sample of `frames` frames with every tag seen, drawn evenly from the
    recordings, and the ends of their rests, by L-BFGS on the robust loss
    over the calibration and every sampled frame's angles. Each recording's
    rests set its angles' zero: every link's mean angle over them is 0.
    With `geometry`, only the cameras are fitted, the geometry held as it
    is. Returns the calibration with the last recording's camera, and one
    for each recording."""
    cal, phis = initial(poses)
    if geometry is not None:
        cal.load_state_dict(geometry.state_dict())
        for p in (cal.pivots, cal.mount_t, cal.mount_rotvec):
            p.requires_grad_(False)
    n = len(poses)
    cam_rv = torch.nn.Parameter(cal.camera_rotvec.detach().expand(n, 3).clone())
    cam_t = torch.nn.Parameter(cal.camera_t.detach().expand(n, 3).clone())
    g = torch.Generator().manual_seed(seed)
    parts = []
    per = max(1, frames // n)
    for k, (p, phi) in enumerate(zip(poses, phis)):
        full = torch.nonzero(p.present.all(-1)).squeeze(-1)
        pick = full[torch.randperm(len(full), generator=g)[:per]]
        ends = torch.from_numpy(rest_ends(p.action))
        ends = ends[p.present[ends].all(-1)]
        ends = ends[torch.randperm(len(ends), generator=g)[: max(per // 4, 50)]]
        for idx, rest in ((pick, False), (ends, True)):
            parts.append((p.position[idx], p.rotation[idx], p.present[idx], phi[idx],
                          torch.full((len(idx),), k), torch.full((len(idx),), rest)))
    position, rotation, present, phi0, rec, is_rest = (torch.cat([x[i] for x in parts]) for i in range(6))
    phi = torch.nn.Parameter(phi0.clone())
    params = [q for q in cal.parameters() if q.requires_grad and q is not cal.camera_rotvec and q is not cal.camera_t]
    opt = torch.optim.LBFGS(params + [cam_rv, cam_t, phi], lr=1, max_iter=20, history_size=50,
                            line_search_fn="strong_wolfe")
    rest_count = torch.zeros(n).index_add_(0, rec[is_rest], torch.ones(int(is_rest.sum()))).clamp(min=1)

    def loss_fn():
        r = cal.residuals(phi, position, rotation, present, cam_rv[rec], cam_t[rec])
        data = huber(r).sum() / present.sum()
        # Gravity: each recording's links' mean angle over its rests' ends is 0.
        mean = torch.zeros(n, NUM_TAGS, dtype=DTYPE).index_add_(0, rec[is_rest], phi[is_rest]) / rest_count[:, None]
        return data + (mean / math.radians(0.01)).pow(2).sum()

    for it in range(iterations // 20):
        def closure():
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        loss = opt.step(closure)
        if log and it % 25 == 0:
            log(f"  fit: step {(it + 1) * 20}, loss {loss.item():.4f}")
    cams = {}
    with torch.no_grad():
        r = cal.residuals(phi, position, rotation, present, cam_rv[rec], cam_t[rec])[~is_rest]
        if log:
            log(f"  fit: loss {loss_fn().item():.4f}; residuals (median |r| in sigmas; across, along, depth, tilt x, "
                "tilt y, yaw), per tag:")
            for i in range(NUM_TAGS):
                log(f"    tag {i}: " + " ".join(f"{v:5.2f}" for v in r[:, i].abs().median(0).values.tolist()))
        for k, p in enumerate(poses):
            c = Calibration()
            c.load_state_dict(cal.state_dict())
            c.camera_rotvec.copy_(cam_rv[k])
            c.camera_t.copy_(cam_t[k])
            cams[p.name] = c
    return cams[poses[-1].name], cams


@torch.no_grad()
def angles(cal: Calibration, p: Poses, iterations: int = 8, chunk: int = 20000) -> tuple[torch.Tensor, torch.Tensor]:
    """Each frame's link angles [T, NUM_TAGS] (rad, 0 hanging) by Gauss-Newton
    on the robust loss from the tags' yaws, and the frame's residuals [T,
    NUM_TAGS, 6] in sigmas. A link none of whose information was seen
    (its own tag, nor any tag further down whose position depends on it)
    keeps the yaw estimate, NaN if its tag was unseen too."""
    R = rotvec_to_matrix(cal.camera_rotvec)
    sign = torch.sign(R[2, 2])
    # Yaw-based start: each tag's yaw less its yaw with the link hanging.
    hang_rot = R @ rotvec_to_matrix(cal.mount_rotvec)
    hang_yaw = tag_yaw(hang_rot)
    out_phi = torch.full(p.present.shape, float("nan"), dtype=DTYPE)
    out_r = torch.zeros(*p.present.shape, 6, dtype=DTYPE)
    for s in range(0, len(p.present), chunk):
        pos, rot, pres = p.position[s:s + chunk], p.rotation[s:s + chunk], p.present[s:s + chunk]
        phi = torch.where(pres, _wrap(sign * (tag_yaw(rot) - hang_yaw)), 0.0)
        for _ in range(iterations):
            r = cal.residuals(phi, pos, rot, pres).reshape(len(phi), -1)
            J = torch.func.vmap(torch.func.jacrev(lambda ph, po, ro, pr: cal.residuals(ph[None], po[None], ro[None], pr[None]).reshape(-1)))(phi, pos, rot, pres)
            # IRLS weights for the Huber loss.
            w = torch.where(r.abs() < HUBER, 1.0, HUBER / r.abs().clamp(min=1e-9))
            JtW = J.transpose(1, 2) * w[:, None, :]
            H = JtW @ J + 1e-6 * torch.eye(NUM_TAGS, dtype=DTYPE)
            phi = phi - torch.linalg.solve(H, (JtW @ r[..., None]).squeeze(-1))
        r = cal.residuals(phi, pos, rot, pres)
        # A link is determined if its tag or any tag below it was seen.
        known = pres.flip(-1).cummax(-1).values.flip(-1)
        out_phi[s:s + chunk] = torch.where(known, _wrap(phi), float("nan"))
        out_r[s:s + chunk] = r
    return out_phi, out_r
