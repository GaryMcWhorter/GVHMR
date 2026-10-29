"""
Compare chained SimpleVO (skips=(1,)) against rotation-averaged SimpleVO on a real video.

There is no ground-truth camera for the demo clips, so two proxies:

1. Consistency: every averaged run also measures skip pairs (i, i+2), (i, i+4), ...
   Those are independent two-view estimates. We score each trajectory by how well it
   agrees with them (median / 90th percentile residual, in degrees). The chained
   trajectory is built from the SAME run's consecutive pairs, so both trajectories are
   judged against identical measurements.
2. Repeatability: pycolmap RANSAC is non-deterministic. Run N times and report how far
   each run's orientations stray from the across-run mean (per frame, and at the last
   keyframe, where chained drift is largest).

Usage:
    python tools/bench/bench_simple_vo_averaging.py [--video PATH] [--runs 5] [--step 8]
           [--skips 1 2 4] [--workers 2]
"""

import argparse
import sys
import time
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

for _name, _attr in [
    ("hmr4d.utils.preproc.tracker", "Tracker"),
    ("hmr4d.utils.preproc.vitfeat_extractor", "Extractor"),
    ("hmr4d.utils.preproc.vitpose", "VitPoseExtractor"),
]:
    if _name not in sys.modules:
        _m = types.ModuleType(_name)
        setattr(_m, _attr, type(_attr, (), {}))
        sys.modules[_name] = _m

import hmr4d.utils.preproc.relpose.simple_vo as simple_vo_mod  # noqa: E402
from hmr4d.utils.preproc.relpose.simple_vo import SimpleVO  # noqa: E402
from hmr4d.utils.preproc.relpose.rotation_averaging import angle_deg  # noqa: E402


def chordal_mean(Rs):
    """(N, 3, 3) -> (3, 3) via SVD projection of the arithmetic mean."""
    U, _, Vt = np.linalg.svd(Rs.mean(0))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    return U @ D @ Vt


def edge_residuals(R, pairs, skip_min=2):
    res = [angle_deg((R[j].T @ T[:3, :3] @ R[i])[None])[0] for (i, j), T in pairs.items() if j - i >= skip_min]
    return np.array(res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="docs/example_video/tennis.mp4")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--skips", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    video = Path(args.video)
    if not video.is_absolute():
        video = _REPO_ROOT / video

    # decode once; SimpleVO.compute() would otherwise re-read the video every run
    frames = simple_vo_mod.read_video_np(str(video), scale=args.scale)
    simple_vo_mod.read_video_np = lambda *a, **k: frames

    chained, averaged, cons_chain, cons_avg, times = [], [], [], [], []
    for r in range(args.runs):
        vo = SimpleVO(str(video), scale=args.scale, step=args.step, num_workers=args.workers, skips=tuple(args.skips))
        t0 = time.perf_counter()
        vo.compute()
        times.append(time.perf_counter() - t0)
        info = vo.last_info
        pairs = info["pairs"]
        n = 1 + max(j for (_, j) in pairs)
        # averaged keyframe rotations (before interpolation): rebuild from the edge-consistent solve
        from hmr4d.utils.preproc.relpose.rotation_averaging import average_rotations

        R_avg = average_rotations(n, {ij: T[:3, :3] for ij, T in pairs.items()})
        R_ch = [np.eye(3)]
        for i in range(n - 1):
            R_ch.append((pairs[(i, i + 1)][:3, :3] if (i, i + 1) in pairs else np.eye(3)) @ R_ch[-1])
        R_ch = np.array(R_ch)
        chained.append(R_ch)
        averaged.append(R_avg)
        cons_chain.append(edge_residuals(R_ch, pairs))
        cons_avg.append(edge_residuals(R_avg, pairs))
        print(
            f"run {r}: {n} keyframes, {len(pairs)} pairs kept, {len(info['dropped_pairs'])} dropped, "
            f"{times[-1]:.1f}s"
        )

    print("\nConsistency with independent skip-pair measurements (deg, lower is better)")
    for name, c in (("chained ", cons_chain), ("averaged", cons_avg)):
        c = np.concatenate(c)
        print(f"  {name}  median {np.median(c):6.3f}   p90 {np.percentile(c, 90):6.3f}   max {c.max():7.3f}")

    print("\nRepeatability across runs (deg from across-run mean, lower is better)")
    for name, runs in (("chained ", chained), ("averaged", averaged)):
        S = np.stack(runs)  # (runs, n, 3, 3)
        dev = np.stack([[angle_deg((S[r, k] @ chordal_mean(S[:, k]).T)[None])[0] for k in range(S.shape[1])]
                        for r in range(S.shape[0])])
        print(f"  {name}  mean {dev.mean():6.3f}   last keyframe {dev[:, -1].mean():6.3f}   max {dev.max():7.3f}")
    print(f"\nmean wall time per run (averaged path, incl. extra pairs): {np.mean(times):.1f}s")


if __name__ == "__main__":
    main()
