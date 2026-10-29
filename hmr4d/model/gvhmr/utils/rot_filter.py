"""
Speed-adaptive smoothing for joint rotations (One-Euro filter on quaternions).

A fixed low-pass either leaves still poses jittery or makes fast motion lag. The One-Euro filter
(Casiez et al. 2012) lowers its cutoff when a signal moves slowly and raises it when it moves
fast. Here it runs on unit quaternions: each step is a slerp from the previous filtered rotation
toward the new sample, so it never leaves the rotation group and needs no renormalized averaging.

Offline (the whole clip is known) it runs forward and then backward over the forward result,
which cancels the filter's lag; online use takes the forward pass alone.

Defaults (min_cutoff 0.1 Hz, beta 2.5) were tuned on synthetic joint motion with ~2 deg jitter
(tools/analysis/rot_filter_synthetic.py), evaluated on held-out seeds.
"""

import math

import torch
import torch.nn.functional as F


def _slerp(q0, q1, t):
    """q0, q1: (..., 4) unit, t: (...,) in [0, 1]. Takes the short way round."""
    d = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(d < 0, -q1, q1)
    d = d.abs().clamp(max=1.0)
    th = torch.acos(d)
    s = torch.sin(th)
    small = s < 1e-6
    t = t[..., None]
    a = torch.where(small, 1 - t, torch.sin((1 - t) * th) / s.clamp(min=1e-12))
    b = torch.where(small, t, torch.sin(t * th) / s.clamp(min=1e-12))
    return F.normalize(a * q0 + b * q1, dim=-1)


def _angle(q0, q1):
    return 2 * torch.acos((q0 * q1).sum(-1).abs().clamp(max=1.0))


def _alpha(cutoff, fps):
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau * fps)


def one_euro_quat(q, fps=30.0, min_cutoff=0.1, beta=2.5, d_cutoff=1.0):
    """
    Args:
        q: (L, ..., 4) unit quaternions over time (first axis)
        min_cutoff: Hz, cutoff when still (lower = steadier)
        beta: how fast the cutoff rises with angular speed (rad/s); higher = less lag
    Returns: (L, ..., 4)
    """
    out = [q[0]]
    speed = torch.zeros(q.shape[1:-1], dtype=q.dtype, device=q.device)
    a_d = _alpha(d_cutoff, fps)
    for t in range(1, q.shape[0]):
        raw_speed = _angle(out[-1], q[t]) * fps
        speed = speed + a_d * (raw_speed - speed)
        cutoff = min_cutoff + beta * speed
        a = 1.0 / (1.0 + fps / (2 * math.pi * cutoff))
        out.append(_slerp(out[-1], q[t], a))
    return torch.stack(out)


def smooth_rotations(q, fps=30.0, min_cutoff=0.1, beta=2.5, zero_lag=True):
    """One-Euro over time; with zero_lag, forward then backward (offline, no lag)."""
    f = one_euro_quat(q, fps, min_cutoff, beta)
    if zero_lag:
        f = one_euro_quat(f.flip(0), fps, min_cutoff, beta).flip(0)
    return f


def smooth_axis_angle(aa, fps=30.0, **kw):
    """(B, L, ..., 3) axis-angle over time axis 1 -> same shape, smoothed."""
    from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle

    q = axis_angle_to_quaternion(aa.double()).transpose(0, 1)  # time first
    q = smooth_rotations(q, fps, **kw)
    return quaternion_to_axis_angle(q.transpose(0, 1)).to(aa)
