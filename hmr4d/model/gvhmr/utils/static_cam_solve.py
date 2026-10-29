"""
Static-camera root trajectory as one sparse least-squares solve.

With a fixed camera, GVHMR has two estimates of where the pelvis is:
  * the in-camera translation (pred_cam -> transl_c): observed every frame, so it does not
    drift, but it is noisy, and depth (from bounding-box scale) is far noisier than the
    image-plane position;
  * the world rollout of predicted local velocities: smooth frame to frame, but it drifts.
It also has per-joint contact probabilities (static_conf) for ankles, feet and wrists.

Every one of those is linear in the unknown pelvis positions once the body's pose is fixed
(joint_w = pelvis_w + offset_w, offsets from FK with zero translation), so the whole thing is a
banded linear least-squares problem:

  camera:     R_w2c p_t + t_w2c  ~=  c_t           per axis, depth down-weighted
  velocity:   p_{t+1} - p_t      ~=  v_t           the rollout's motion
  contact:    joint j still while in contact       (p_{t+1}+o_{t+1,j}) - (p_t+o_{t,j}) ~= 0
  ground:     contacting foot joints at height g   (p_t + o_{t,j})_y ~= g
  smooth:     p_{t+1} - 2 p_t + p_{t-1} ~= 0

Weights are standard deviations in meters (per frame where it applies). Defaults were tuned on
synthetic walkers with held-out evaluation (tools/analysis/static_cam_synthetic.py); real network
error statistics may want different values.

Unknowns: p (L x 3), camera position t_w2c (3), ground height g. A weak prior on p_0 fixes the
global gauge. The result is shifted so the ground sits at y = 0. World is y-up (GVHMR "ay").
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def chordal_mean(Rs):
    """(N, 3, 3) -> (3, 3) nearest rotation to the arithmetic mean."""
    U, _, Vt = np.linalg.svd(np.asarray(Rs).mean(0))
    return U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt


def solve_static_cam_root(
    pelvis_c,
    offsets_w,
    vel_w,
    contact_prob,
    contact_joints,
    R_c2w,
    foot_joints=(),
    sigma_cam_lat=0.02,
    sigma_cam_depth=0.15,
    sigma_vel=0.004,
    sigma_contact=0.002,
    sigma_ground=0.01,
    sigma_smooth=0.002,
    contact_thr=0.5,
    p0_prior=None,
):
    """
    Args:
        pelvis_c: (L, 3) pelvis position in camera coords (from the in-camera prediction)
        offsets_w: (L, J, 3) joint positions relative to the pelvis, world coords (FK, zero transl)
        vel_w: (L, 3) pelvis world displacement from frame t to t+1 (rollout); last row unused
        contact_prob: (L, C) probability joint contact_joints[c] is planted at frame t
        contact_joints: length-C indices into J
        R_c2w: (3, 3) static camera-to-world rotation (world y-up)
        foot_joints: indices (into contact_joints positions) that touch the ground when planted
        p0_prior: (3,) pelvis world position at frame 0 to pin the gauge (default origin)
    Returns:
        p: (L, 3) pelvis world positions, ground at y = 0
        info: dict with camera position, ground height before the shift, residual stats
    """
    pelvis_c = np.asarray(pelvis_c, np.float64)
    offsets_w = np.asarray(offsets_w, np.float64)
    vel_w = np.asarray(vel_w, np.float64)
    contact_prob = np.asarray(contact_prob, np.float64)
    L = len(pelvis_c)
    R_w2c = np.asarray(R_c2w, np.float64).T
    n_p, i_t, i_g = 3 * L, 3 * L, 3 * L + 3
    n = 3 * L + 4

    rows, cols, vals, rhs = [], [], [], []
    r = [0]  # running row counter

    def add(entries, b, w):
        """entries: list of (col, coeff) for one residual row."""
        for c, v in entries:
            rows.append(r[0]); cols.append(c); vals.append(v * w)
        rhs.append(b * w)
        r[0] += 1

    P = lambda t, k: 3 * t + k  # column of p_t[k]

    # camera term, in camera axes: (R_w2c p_t + t) - c_t
    w_axis = np.array([1 / sigma_cam_lat, 1 / sigma_cam_lat, 1 / sigma_cam_depth])
    for t in range(L):
        for a in range(3):
            ents = [(P(t, k), R_w2c[a, k]) for k in range(3)] + [(i_t + a, 1.0)]
            add(ents, pelvis_c[t, a], w_axis[a])

    # velocity term
    for t in range(L - 1):
        for k in range(3):
            add([(P(t + 1, k), 1.0), (P(t, k), -1.0)], vel_w[t, k], 1 / sigma_vel)

    # contact: planted joint does not move (weight: both frames' probabilities)
    for ci, j in enumerate(contact_joints):
        pc = np.minimum(contact_prob[:-1, ci], contact_prob[1:, ci])
        for t in np.nonzero(pc > contact_thr)[0]:
            w = pc[t] / sigma_contact
            d_off = offsets_w[t + 1, j] - offsets_w[t, j]
            for k in range(3):
                add([(P(t + 1, k), 1.0), (P(t, k), -1.0)], -d_off[k], w)

    # ground: planted foot joints at height g
    for ci in foot_joints:
        j = contact_joints[ci]
        for t in np.nonzero(contact_prob[:, ci] > contact_thr)[0]:
            add([(P(t, 1), 1.0), (i_g, -1.0)], -offsets_w[t, j, 1], contact_prob[t, ci] / sigma_ground)

    # smoothness
    for t in range(1, L - 1):
        for k in range(3):
            add([(P(t + 1, k), 1.0), (P(t, k), -2.0), (P(t - 1, k), 1.0)], 0.0, 1 / sigma_smooth)

    # gauge
    p0 = np.zeros(3) if p0_prior is None else np.asarray(p0_prior, np.float64)
    for k in range(3):
        add([(P(0, k), 1.0)], p0[k], 1e-2)

    A = sp.csr_matrix((vals, (rows, cols)), shape=(r[0], n))
    b = np.asarray(rhs)
    H = (A.T @ A + 1e-10 * sp.eye(n)).tocsc()
    x = spla.spsolve(H, A.T @ b)
    p = x[:n_p].reshape(L, 3)
    t_w2c, g = x[i_t:i_t + 3], x[i_g]
    if not len(foot_joints) or not (contact_prob[:, list(foot_joints)] > contact_thr).any():
        g = (p[:, None, 1] + offsets_w[..., 1]).min()  # no contacts: lowest joint is the floor
    p[:, 1] -= g
    res = A @ x - b
    return p, {"t_w2c": t_w2c, "ground_y": g, "rms_residual": float(np.sqrt(np.mean(res**2)))}
