"""
Synthetic regression test for hmr4d/utils/preproc/relpose/rotation_averaging.py.
No video or GPU needed. Exits non-zero on failure.

    python tools/bench/test_rotation_averaging.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

_p = Path(__file__).resolve().parents[2] / "hmr4d/utils/preproc/relpose/rotation_averaging.py"
_spec = importlib.util.spec_from_file_location("rotation_averaging", _p)
ra = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ra)


def make_graph(rng, n, noise_deg, outlier_p, fail_p, skips=(1, 2, 4)):
    true = [np.eye(3)]
    for _ in range(n - 1):
        true.append(ra.so3_exp(rng.normal(0, np.radians(3), 3))[0] @ true[-1])
    true = np.array(true)
    edges = {}
    for k in skips:
        for i in range(n - k):
            if rng.random() < fail_p:
                continue
            e = rng.normal(0, np.radians(noise_deg), 3)
            if rng.random() < outlier_p:
                e = rng.normal(0, np.radians(15), 3)
            edges[(i, i + k)] = ra.so3_exp(e)[0] @ true[i + k] @ true[i].T
    return true, edges


def err_deg(R, true):
    """Per-keyframe error after removing the global gauge (R_k <- R_k G).

    GVHMR consumes only relative camera rotations (cam_angvel), so a constant offset is
    harmless. It does matter here: if the anchor keyframe's own pairs are mostly bad, the
    solver rightly keeps every other keyframe self-consistent and leaves only the anchor
    (one frame step) wrong, which raw error would report as a 15 deg error everywhere.
    """
    U, _, Vt = np.linalg.svd(np.einsum("aji,ajk->ik", R, true))
    G = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return ra.angle_deg(np.einsum("aij,jk,alk->ail", R, G, true))


def chain(n, edges):
    R = [np.eye(3)]
    for i in range(n - 1):
        R.append(edges.get((i, i + 1), np.eye(3)) @ R[-1])
    return np.array(R)


def main():
    rng = np.random.default_rng(0)
    ok = True

    # exact data is recovered exactly, and keyframe 0 stays the identity
    true, edges = make_graph(rng, 60, 0.0, 0.0, 0.0)
    R = ra.average_rotations(60, edges)
    e = err_deg(R, true).max()
    ok &= e < 1e-6 and np.allclose(R[0], np.eye(3))
    print(f"{'PASS' if e < 1e-6 else 'FAIL'}  noiseless recovery  max err {e:.2e} deg")

    # noise only: averaging beats chaining on drift
    C, A = [], []
    for _ in range(10):
        true, edges = make_graph(rng, 225, 0.3, 0.0, 0.0)
        C.append(err_deg(chain(225, edges), true)[-1])
        A.append(err_deg(ra.average_rotations(225, edges), true)[-1])
    good = np.mean(A) < 0.5 * np.mean(C)
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  noise: final drift chained {np.mean(C):.2f} -> averaged {np.mean(A):.2f} deg")

    # bad pairs, including on the first edges (the hardest place to outvote them)
    worst = []
    for t in range(60):
        true, edges = make_graph(rng, 40, 0.3, 0.03, 0.0)
        if t % 3 == 0:  # force a bad pair at the anchor
            edges[(0, 1)] = ra.so3_exp(np.radians([15.0, -10.0, 5.0]))[0] @ edges[(0, 1)]
        # if most of the anchor's own pairs are bad, the anchor alone ends up wrong (one bad
        # cam_angvel step, same as chaining); judge the remaining keyframes
        worst.append(err_deg(ra.average_rotations(40, edges)[1:], true[1:]).max())
    good = np.max(worst) < 3.0
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  3% bad pairs (+ forced bad first pair): worst keyframe err {np.max(worst):.2f} deg")

    # missing pairs, including a keyframe with no incoming edge at all
    true, edges = make_graph(rng, 40, 0.3, 0.0, 0.0)
    for k in (1, 2, 4):
        edges.pop((20 - k, 20), None)
    R, info = ra.average_rotations(40, edges, return_info=True)
    e = err_deg(R, true).max()
    good = e < 3.0 and info["n_unconstrained"] == 1
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  keyframe with no incoming pairs: max err {e:.2f} deg")

    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
