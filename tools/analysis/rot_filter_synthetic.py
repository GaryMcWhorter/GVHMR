"""
Synthetic test for joint-rotation smoothing: still holds, slow sway and fast swings, with
network-like jitter (about 2 deg per frame, partly correlated). Compares no smoothing, a fixed
Gaussian, and the One-Euro quaternion filter (forward only, and forward-backward).

    PYTHONPATH=<pytorch3d shim>:. python tools/analysis/rot_filter_synthetic.py
"""
import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hmr4d.model.gvhmr.utils.rot_filter import _angle, _slerp, smooth_rotations, one_euro_quat  # noqa: E402
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle  # noqa: E402

FPS = 30


def make_case(rng, J=8):
    """(L, J, 4) truth and noisy quaternions, plus a per-frame 'fast' mask."""
    segs = [(2.0, "still"), (2.0, "slow"), (1.0, "fast"), (1.5, "still"), (1.5, "fast"), (2.0, "slow"), (1.0, "still")]
    L = sum(int(d * FPS) for d, _ in segs)
    aa = np.zeros((L, J, 3))
    fast = np.zeros(L, bool)
    t0 = 0
    base = rng.normal(0, 0.4, (J, 3))
    amp = rng.uniform(0.3, 1.2, (J, 3))
    phase = rng.uniform(0, 2 * np.pi, (J, 3))
    state = base.copy()
    for d, kind in segs:
        n = int(d * FPS)
        tt = np.arange(n) / FPS
        f = {"still": 0.0, "slow": 0.4, "fast": 2.0}[kind]
        for i in range(n):
            if kind != "still":
                state = base + amp * np.sin(2 * np.pi * f * (t0 + i) / FPS + phase) * (1.0 if kind == "fast" else 0.5)
            aa[t0 + i] = state
        fast[t0:t0 + n] = kind == "fast"
        t0 += n
    aa = gaussian_filter1d(aa, 2, axis=0)  # no instant velocity changes
    q = axis_angle_to_quaternion(torch.from_numpy(aa))
    noise = rng.normal(0, 1, aa.shape) + gaussian_filter1d(rng.normal(0, 1, aa.shape), 2, axis=0) * 2.5
    noise *= math.radians(2.0) / np.sqrt(2)
    qn = axis_angle_to_quaternion(torch.from_numpy(aa + noise))
    return q, qn, fast


def score(est, q, fast):
    err = torch.rad2deg(_angle(est, q)).numpy()  # (L, J)
    # jitter: frame-to-frame change beyond the truth's own change
    d_est = torch.rad2deg(_angle(est[1:], est[:-1])).numpy()
    d_true = torch.rad2deg(_angle(q[1:], q[:-1])).numpy()
    return {
        "err_deg": err.mean(),
        "err_still": err[~fast].mean(),
        "err_fast": err[fast].mean(),
        "jitter_deg": np.abs(d_est - d_true)[~fast[1:]].mean(),
    }


def gaussian(qn, sigma):
    aa = quaternion_to_axis_angle(qn).numpy()
    return axis_angle_to_quaternion(torch.from_numpy(gaussian_filter1d(aa, sigma, axis=0)))


def run(seeds, methods):
    rows = {}
    for name, fn in methods.items():
        acc = []
        for s in seeds:
            q, qn, fast = make_case(np.random.default_rng(s))
            acc.append(score(fn(qn), q, fast))
        rows[name] = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
    return rows


def main():
    # tune One-Euro on seeds 100-109
    best = None
    for mc, beta in itertools.product([0.1, 0.2, 0.3, 0.6, 1.0], [0.3, 0.6, 1.2, 2.5, 5.0]):
        r = run(range(100, 110), {"x": lambda qn: smooth_rotations(qn, FPS, mc, beta)})["x"]
        if best is None or r["err_deg"] < best[0]["err_deg"]:
            best = (r, mc, beta)
    _, mc, beta = best
    print(f"tuned on seeds 100-109: min_cutoff={mc} Hz, beta={beta}\n")
    methods = {
        "none": lambda qn: qn,
        "gaussian s=2": lambda qn: gaussian(qn, 2),
        "gaussian s=4": lambda qn: gaussian(qn, 4),
        "1euro fwd": lambda qn: one_euro_quat(qn, FPS, mc, beta),
        "1euro zero-lag": lambda qn: smooth_rotations(qn, FPS, mc, beta),
    }
    rows = run(range(10), methods)
    keys = list(next(iter(rows.values())))
    print(f"{'':16s}" + "".join(f"{k:>12s}" for k in keys))
    for n, r in rows.items():
        print(f"{n:16s}" + "".join(f"{r[k]:12.3f}" for k in keys))


if __name__ == "__main__":
    main()
