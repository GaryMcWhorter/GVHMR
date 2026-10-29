"""
Ground-truth test for SimpleVO: render a camera rotating inside a textured sphere.

A pure rotation of a pinhole camera is exactly what this renders (every pixel looks up
the environment along its ray), so the true R_w2c of every frame is known. The frames
go through the real SimpleVO pipeline (SIFT + pycolmap), chained and averaged, several
runs each, and both are scored against the truth.

    python tools/analysis/virtual_camera_vo.py --out vo_compare.json
"""

import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
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
from hmr4d.utils.preproc.relpose.rotation_averaging import angle_deg, so3_exp  # noqa: E402
from hmr4d.utils.preproc.relpose.utils import focal_length_from_mm  # noqa: E402


class SphereEnv:
    """Procedural environment: multi-octave 3D value noise of the ray direction, with
    thresholded blobs and stripes so SIFT has corners at several scales."""

    def __init__(self, seed=0, n=64):
        rng = np.random.default_rng(seed)
        self.n = n
        self.lat = [rng.random((n, n, n)).astype(np.float32) for _ in range(3)]

    def _noise(self, p, lat):
        n = self.n
        p0 = np.floor(p).astype(np.int64)
        f = (p - p0).astype(np.float32)
        f = f * f * (3 - 2 * f)
        out = 0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = (f[..., 0] if dx else 1 - f[..., 0]) * (f[..., 1] if dy else 1 - f[..., 1]) * (
                        f[..., 2] if dz else 1 - f[..., 2]
                    )
                    out = out + w * lat[(p0[..., 0] + dx) % n, (p0[..., 1] + dy) % n, (p0[..., 2] + dz) % n]
        return out

    def color(self, d):
        """d: (..., 3) unit directions -> (..., 3) uint8"""
        p = d * 6.0 + 32.0
        a = self._noise(p, self.lat[0]) * 0.6 + self._noise(p * 2.7, self.lat[1]) * 0.3
        b = self._noise(p * 7.3, self.lat[2])
        blobs = (self._noise(p * 1.9, self.lat[1]) > 0.62).astype(np.float32)
        stripes = (np.sin(d[..., 2] * 40 + a * 6) > 0.7).astype(np.float32) * 0.25
        v = np.clip(0.15 + 0.6 * a + 0.25 * b - 0.35 * blobs + stripes, 0, 1)
        rgb = np.stack([v, v * 0.9 + 0.08 * blobs, v * 0.75 + 0.2 * (1 - a)], -1)
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def euler_R_w2c(yaw, pitch, roll):
    """World z-up; camera looks along +x at yaw=pitch=0 with image-down = -z."""
    base = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0.0]])
    Rz = so3_exp(np.array([[0, 0, yaw]]))[0]
    Rx = so3_exp(np.array([[-pitch, 0, 0]]))[0]
    Rr = so3_exp(np.array([[0, 0, roll]]))[0]
    R_c2w = Rz @ base.T @ Rx @ Rr
    return R_c2w.T


def render(env, Rs, W, H, f):
    u, v = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    rays_c = np.stack([(u - W / 2) / f, (v - H / 2) / f, np.ones_like(u)], -1)
    rays_c /= np.linalg.norm(rays_c, axis=-1, keepdims=True)
    frames = np.empty((len(Rs), H, W, 3), np.uint8)
    for t, R in enumerate(Rs):
        frames[t] = env.color(rays_c @ R)  # world dir = R^T d_c  ->  row-vector form d_c @ R
    return frames


def trajectory(L, fps=30):
    """Handheld walk-and-pan: 150 deg of yaw, pitch drifting down, wobble on all axes."""
    t = np.arange(L) / fps
    yaw = np.radians(150) * (t / t[-1]) ** 1.2 + np.radians(2.0) * np.sin(2 * np.pi * 0.7 * t)
    pitch = np.radians(10 + 15 * t / t[-1]) + np.radians(1.5) * np.sin(2 * np.pi * 1.3 * t + 1)
    roll = np.radians(2.5) * np.sin(2 * np.pi * 0.9 * t + 2)
    return np.stack([euler_R_w2c(*a) for a in zip(yaw, pitch, roll)]), np.degrees(yaw)


def heading_deg(R_w2c):
    """Camera view axis heading in the world ground plane."""
    z_w = R_w2c[:, 2, :]  # camera z axis in world = third row of R_w2c
    return np.degrees(np.unwrap(np.arctan2(z_w[:, 1], z_w[:, 0])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--step", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", default="vo_compare.json")
    ap.add_argument("--save_frames", default=None, help="optional .npz of a few frames for previews")
    args = ap.parse_args()

    W, H = 406, 360
    f = focal_length_from_mm(W, H, 24)
    R_true, _ = trajectory(args.frames)
    print(f"rendering {args.frames} frames at {W}x{H} ...")
    frames = render(SphereEnv(), R_true, W, H, f)
    if args.save_frames:
        idx = np.linspace(0, args.frames - 1, 6).astype(int)
        np.savez_compressed(args.save_frames, idx=idx, frames=frames[idx])
    simple_vo_mod.read_video_np = lambda *a, **k: frames

    def score(T_list):
        R = np.stack([np.asarray(T)[:3, :3] for T in T_list])
        G = R[0].T @ R_true[0]  # align frame 0 (VO starts at identity)
        R_al = R @ G
        err = angle_deg(np.einsum("aij,akj->aik", R_al, R_true))
        return R_al, err

    out = {"frames": args.frames, "fps": 30, "step": args.step, "true_heading": heading_deg(R_true).tolist(), "runs": []}
    for r in range(args.runs):
        for mode, skips in (("chained", (1,)), ("averaged", (1, 2, 4))):
            vo = SimpleVO("virtual", step=args.step, num_workers=args.workers, skips=skips)
            R_al, err = score(vo.compute())
            out["runs"].append(
                {"run": r, "mode": mode, "heading": heading_deg(R_al).tolist(), "err": err.tolist()}
            )
            print(f"run {r} {mode:8s}: mean err {err.mean():.3f}  final {err[-1]:.3f}  max {err.max():.3f} deg")
    Path(args.out).write_text(json.dumps(out))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
