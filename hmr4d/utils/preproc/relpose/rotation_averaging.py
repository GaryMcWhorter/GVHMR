"""
Robust rotation averaging for SimpleVO keyframes.

Chaining keyframe-to-keyframe rotations lets RANSAC noise random-walk, and a
single bad pair rotates every later frame. With redundant pairs (i -> i+1,
i -> i+2, i -> i+4, ...) each keyframe gets several independent estimates:

1. Initialization: each keyframe takes the medoid of the orientations its
   edges propose (incoming in a forward pass, then incoming and outgoing in a
   few sweeps), so one bad pair is outvoted instead of propagated.
2. Refinement: IRLS Gauss-Newton on SO(3) (Lie-algebraic averaging, Govindu
   2004 style) with Cauchy weights, solving all keyframes jointly.

Convention matches SimpleVO: an edge (i, j) carries R_ij with
R_w2c[j] = R_ij @ R_w2c[i]. Keyframe 0 is fixed to identity.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _hat(w):
    """(N, 3) -> (N, 3, 3)"""
    O = np.zeros((len(w), 3, 3))
    O[:, 0, 1], O[:, 0, 2] = -w[:, 2], w[:, 1]
    O[:, 1, 0], O[:, 1, 2] = w[:, 2], -w[:, 0]
    O[:, 2, 0], O[:, 2, 1] = -w[:, 1], w[:, 0]
    return O


def so3_exp(w):
    """(N, 3) axis-angle -> (N, 3, 3)"""
    w = np.atleast_2d(w)
    th = np.linalg.norm(w, axis=-1)[:, None, None]
    K = _hat(w)
    small = th < 1e-8
    th_s = np.where(small, 1.0, th)
    A = np.where(small, 1.0, np.sin(th_s) / th_s)
    B = np.where(small, 0.5, (1 - np.cos(th_s)) / th_s**2)
    return np.eye(3) + A * K + B * K @ K


def so3_log(R):
    """(N, 3, 3) -> (N, 3) axis-angle (accurate near 0; near pi falls back to the symmetric part)"""
    R = np.asarray(R).reshape(-1, 3, 3)
    c = np.clip((np.trace(R, axis1=1, axis2=2) - 1) / 2, -1, 1)
    th = np.arccos(c)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], -1)
    s = np.sin(th)
    out = np.where((th < 1e-6)[:, None], 0.5 * v, v * (th / (2 * np.maximum(s, 1e-12)))[:, None])
    near_pi = th > np.pi - 1e-3
    if near_pi.any():  # axis from the largest column of (R + I)/2
        for k in np.nonzero(near_pi)[0]:
            M = (R[k] + np.eye(3)) / 2
            col = M[:, np.argmax(np.diag(M))]
            out[k] = col / np.linalg.norm(col) * th[k]
    return out


def angle_deg(R):
    return np.degrees(np.linalg.norm(so3_log(R), axis=-1))


def _medoid(cands):
    if len(cands) == 1:
        return cands[0]
    C = np.stack(cands)
    D = angle_deg(np.einsum("aij,bkj->abik", C, C).reshape(-1, 3, 3)).reshape(len(C), len(C))
    return C[int(np.argmin(D.sum(1)))]


def average_rotations(n, edges, cauchy_deg=2.0, max_iters=30, tol=1e-9, init_sweeps=3, return_info=False):
    """
    Args:
        n: number of keyframes
        edges: dict {(i, j): R_ij (3, 3)} with i < j; failed pairs simply omitted
    Returns:
        R: (n, 3, 3) R_w2c per keyframe, R[0] = I
        info (optional): dict with per-edge residuals in degrees and final weights
    """
    edges = {k: np.asarray(v, dtype=np.float64) for k, v in edges.items()}
    incoming = {j: [] for j in range(n)}
    for (i, j) in edges:
        incoming[j].append(i)

    # 1. robust init: a forward pass using incoming edges, then sweeps where every keyframe
    #    re-votes using all its edges (incoming and outgoing). The sweeps matter near the start
    #    of the clip, where a keyframe has fewer than 3 incoming edges and can't outvote one bad
    #    pair on its own; its outgoing edges supply the extra votes.
    outgoing = {i: [] for i in range(n)}
    for (i, j) in edges:
        outgoing[i].append(j)
    R = np.zeros((n, 3, 3))
    R[0] = np.eye(3)
    n_unconstrained = 0
    for j in range(1, n):
        cands = [edges[(i, j)] @ R[i] for i in incoming[j]]
        if not cands:  # every pair into j failed: hold the previous orientation
            cands = [R[j - 1]]
            n_unconstrained += 1
        R[j] = _medoid(cands)
    for _ in range(init_sweeps):
        changed = 0.0
        for j in list(range(n)) + list(range(n - 1, -1, -1)):
            cands = [edges[(i, j)] @ R[i] for i in incoming[j]]
            cands += [edges[(j, k)].T @ R[k] for k in outgoing[j]]
            if not cands:
                continue
            new = _medoid(cands)
            changed = max(changed, angle_deg((new @ R[j].T)[None])[0])
            R[j] = new
        if changed < 1e-6:
            break
    # Keyframe 0 votes too (it is the anchor only by convention, and with few edges it is the most
    # likely to sit on the wrong side of a bad pair). Re-gauge so keyframe 0 is the identity again:
    # R_j <- R_j R_0^T leaves every R_j R_i^T, and so every edge residual, unchanged.
    R = R @ R[0].T

    keys = list(edges)
    if not keys or n < 2:
        return (R, {"n_unconstrained": n_unconstrained}) if return_info else R
    I_idx = np.array([k[0] for k in keys])
    J_idx = np.array([k[1] for k in keys])
    Rij = np.stack([edges[k] for k in keys])
    m, nv = len(keys), 3 * (n - 1)
    c = np.radians(cauchy_deg)

    def residuals(R):
        return so3_log(np.einsum("eji,ejk,ekl->eil", R[J_idx], Rij, R[I_idx]))  # log(R_j^T R_ij R_i)

    # 2. IRLS Gauss-Newton with right (body) perturbations R_k <- R_k exp(d_k)
    for _ in range(max_iters):
        r = residuals(R)
        w = 1.0 / (1.0 + (np.linalg.norm(r, axis=-1) / c) ** 2)
        sw = np.sqrt(w)[:, None, None]
        rows, cols, vals = [], [], []
        blk_I = np.broadcast_to(np.eye(3), (m, 3, 3)) * sw
        # r = log(R_j^T R_ij R_i): perturbing R_j by exp(d) moves r by about -d, R_i by about +d
        blk_i = -blk_I
        r3 = (np.arange(m) * 3)[:, None, None] + np.arange(3)[None, :, None]
        for blk, node in ((blk_I, J_idx), (blk_i, I_idx)):
            keep = node > 0
            c3 = ((node - 1) * 3)[:, None, None] + np.arange(3)[None, None, :]
            rows.append(np.broadcast_to(r3, (m, 3, 3))[keep].ravel())
            cols.append(np.broadcast_to(c3, (m, 3, 3))[keep].ravel())
            vals.append(blk[keep].ravel())
        A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * m, nv))
        b = (r * np.sqrt(w)[:, None]).ravel()
        H = (A.T @ A + 1e-9 * sp.eye(nv)).tocsc()
        d = spla.spsolve(H, A.T @ b).reshape(-1, 3)
        R[1:] = R[1:] @ so3_exp(d)
        if np.abs(d).max() < tol:
            break

    if not return_info:
        return R
    r = residuals(R)
    return R, {
        "edge_residual_deg": dict(zip(keys, np.degrees(np.linalg.norm(r, axis=-1)))),
        "edge_weight": dict(zip(keys, 1.0 / (1.0 + (np.linalg.norm(r, axis=-1) / c) ** 2))),
        "n_unconstrained": n_unconstrained,
    }
