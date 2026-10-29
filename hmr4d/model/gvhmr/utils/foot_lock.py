"""
Foot locking for GVHMR output: decide where planted feet should stay, then bend the legs to
put them there.

1. contact_lock_targets: turn per-frame contact probabilities into contact intervals (with
   hysteresis, so a flickering probability doesn't make the foot stutter), give each interval
   one fixed world position (a probability-weighted median of where the network put the joint),
   optionally pin ground joints to the floor, and ease in and out of each lock so the foot
   doesn't pop at touchdown or lift-off.
2. two_bone_ik: analytic hip-knee-ankle IK on GVHMR's local transform stack, keeping the knee
   in its current bend plane, followed by an ankle swing that points the foot at its toe target.

World is y-up with the floor at y = 0 (the global output after post-processing).
"""

import numpy as np
import torch
import torch.nn.functional as F

import hmr4d.utils.matrix as matrix


def contact_intervals(prob, on=0.6, off=0.4, min_len=3):
    """prob: (L,) -> list of (start, end) inclusive, using hysteresis thresholds."""
    out, active, s = [], False, 0
    for t, p in enumerate(prob):
        if not active and p >= on:
            active, s = True, t
        elif active and p < off:
            active = False
            if t - s >= min_len:
                out.append((s, t - 1))
    if active and len(prob) - s >= min_len:
        out.append((s, len(prob) - 1))
    return out


def _weighted_median(x, w):
    o = np.argsort(x)
    c = np.cumsum(w[o])
    return x[o][np.searchsorted(c, 0.5 * c[-1])]


def contact_lock_targets(pos_w, prob, ground=(), floor_y=0.0, on=0.6, off=0.4, min_len=3, blend=4):
    """
    Args:
        pos_w: (L, C, 3) world positions of the contact joints after root post-processing
        prob: (L, C) contact probabilities
        ground: indices into C of joints that rest on the floor when planted (toes)
        blend: frames to ease into and out of each lock
    Returns:
        target: (L, C, 3) where each joint should be
        weight: (L, C) 1 inside a lock, easing to 0 outside (0 = leave the joint alone)
    """
    pos_w = np.asarray(pos_w, np.float64)
    L, C, _ = pos_w.shape
    target = pos_w.copy()
    weight = np.zeros((L, C))
    for c in range(C):
        for s, e in contact_intervals(prob[:, c], on, off, min_len):
            w = np.clip(prob[s:e + 1, c], 1e-3, None)
            lock = np.array([_weighted_median(pos_w[s:e + 1, c, k], w) for k in range(3)])
            if c in ground:
                lock[1] = floor_y
            target[s:e + 1, c] = lock
            weight[s:e + 1, c] = 1.0
            for k in range(1, blend + 1):  # ease: the target slides from the free path onto the lock
                a = 0.5 + 0.5 * np.cos(np.pi * k / (blend + 1))  # 1 -> 0
                for t in (s - k, e + k):
                    if 0 <= t < L and a > weight[t, c]:
                        target[t, c] = a * lock + (1 - a) * pos_w[t, c]
                        weight[t, c] = a
    return target, weight


def _rot_between(a, b, eps=1e-8):
    """Shortest-arc rotation matrices taking unit vectors a to b. (..., 3) -> (..., 3, 3)"""
    a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
    v = torch.cross(a, b, dim=-1)
    c = (a * b).sum(-1, keepdim=True)
    q = torch.cat([1 + c, v], -1)  # quaternion (w, xyz) of the rotor (1 + b a)
    bad = (1 + c[..., 0]) < eps
    if bad.any():  # opposite vectors: any perpendicular axis, angle pi
        perp = torch.cross(a, torch.tensor([1.0, 0, 0]).to(a).expand_as(a), dim=-1)
        perp2 = torch.cross(a, torch.tensor([0, 1.0, 0]).to(a).expand_as(a), dim=-1)
        perp = torch.where(perp.norm(dim=-1, keepdim=True) > 1e-3, perp, perp2)
        q = torch.where(bad[..., None], torch.cat([torch.zeros_like(c), F.normalize(perp, dim=-1)], -1), q)
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1).reshape(q.shape[:-1] + (3, 3))


def _set_global_rot(local_mat, glob_mat, parents, j, R_new_global):
    """Change joint j's local rotation so its global rotation becomes R_new_global."""
    R_parent = glob_mat[..., parents[j], :3, :3]
    local_mat[..., j, :3, :3] = R_parent.mT @ R_new_global


def two_bone_ik(local_mat, parents, hip, knee, ankle, toe, ankle_target, ankle_w, toe_target=None, toe_w=None):
    """
    Analytic leg IK on GVHMR's (B, L, J, 4, 4) local transforms (translation already applied to
    the root). Moves hip and knee so the ankle reaches ankle_target, keeping the leg's bend plane;
    then swings the ankle so the toe points at toe_target. Blend weights in [0, 1] per frame.
    Returns a new local_mat.
    """
    local_mat = local_mat.clone()
    eps = 1e-6
    glob = matrix.forward_kinematics(local_mat, parents)
    P = lambda j: glob[..., j, :3, 3]
    p_h, p_k, p_a = P(hip), P(knee), P(ankle)
    tgt = p_a + ankle_w[..., None] * (ankle_target - p_a)
    l1 = (p_k - p_h).norm(dim=-1)
    l2 = (p_a - p_k).norm(dim=-1)
    d = (tgt - p_h).norm(dim=-1).clamp(min=eps)
    d_c = d.clamp(max=(l1 + l2) * 0.9999, min=(l1 - l2).abs() + 1e-4)

    # knee interior angle: current and desired (law of cosines)
    def interior(a, b, c):
        return torch.acos(((a * a + b * b - c * c) / (2 * a * b)).clamp(-1, 1))

    cur = interior(l1, l2, (p_a - p_h).norm(dim=-1))
    want = interior(l1, l2, d_c)
    # bend axis: normal of the hip-knee-ankle plane (fallback when the leg is straight)
    n = torch.cross(p_k - p_h, p_a - p_k, dim=-1)
    straight = n.norm(dim=-1, keepdim=True) < 1e-6
    fallback = torch.cross(p_a - p_h, torch.tensor([0.0, 1, 0]).to(n).expand_as(n), dim=-1)
    n = F.normalize(torch.where(straight, fallback, n), dim=-1)
    # rotate the knee about the bend axis by (cur - want): opens or closes the knee
    ang = (cur - want)
    K = torch.zeros(n.shape[:-1] + (3, 3)).to(n)
    K[..., 0, 1], K[..., 0, 2] = -n[..., 2], n[..., 1]
    K[..., 1, 0], K[..., 1, 2] = n[..., 2], -n[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -n[..., 1], n[..., 0]
    s, c = torch.sin(ang)[..., None, None], torch.cos(ang)[..., None, None]
    R_bend = torch.eye(3).to(n) + s * K + (1 - c) * K @ K
    R_knee_g = glob[..., knee, :3, :3]
    R_ankle_g = glob[..., ankle, :3, :3]
    _set_global_rot(local_mat, glob, parents, knee, R_bend @ R_knee_g)

    # swing the hip so hip->ankle points at the target
    glob = matrix.forward_kinematics(local_mat, parents)
    p_a2 = glob[..., ankle, :3, 3]
    R_swing = _rot_between(p_a2 - p_h, tgt - p_h)
    _set_global_rot(local_mat, glob, parents, hip, R_swing @ glob[..., hip, :3, :3])

    # keep the ankle's global orientation as it was (feet don't twist because the leg moved)
    glob = matrix.forward_kinematics(local_mat, parents)
    _set_global_rot(local_mat, glob, parents, ankle, R_ankle_g)

    # toe: swing the ankle so ankle->toe points at the toe target
    if toe_target is not None:
        glob = matrix.forward_kinematics(local_mat, parents)
        p_a3, p_t = glob[..., ankle, :3, 3], glob[..., toe, :3, 3]
        tt = p_t + toe_w[..., None] * (toe_target - p_t)
        R_t = _rot_between(p_t - p_a3, tt - p_a3)
        _set_global_rot(local_mat, glob, parents, ankle, R_t @ glob[..., ankle, :3, :3])
    return local_mat
