"""
Synthetic ground-truth test for static-camera root trajectory post-processing (webcam case).

A procedural walker (pelvis path, planted footsteps, swing arcs) is seen by a fixed camera.
GVHMR-like predictions are made from the truth with realistic error:
  * in-camera pelvis: 1.5 cm image-plane noise, 8 cm depth noise, and a 4% scale bias
    (monocular depth from a bounding box can't tell a big person far away from a small one near)
  * world rollout velocity: 4 deg heading bias, 7% speed bias, 4 mm/frame noise (it drifts)
  * joint offsets from FK: 1 cm noise
  * per-frame camera-to-world rotation estimate: 1.5 deg noise
  * contact probabilities: truth blurred, with 4% of frames flipped
Three post-processes are scored against the truth:
  raw     the world rollout, dropped to the floor (what GVHMR outputs without post-processing)
  legacy  a faithful port of pp_static_joint_cam
  lsq     solve_static_cam_root (this branch)

    python tools/analysis/static_cam_synthetic.py [--seeds 5] [--out static_cam.json]
"""

import argparse
import sys
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

_p = Path(__file__).resolve().parents[2] / "hmr4d/model/gvhmr/utils/static_cam_solve.py"
_spec = importlib.util.spec_from_file_location("static_cam_solve", _p)
scs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scs)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hmr4d.model.gvhmr.utils.foot_lock import contact_lock_targets  # noqa: E402

FPS = 30
# joint layout used here: 0 pelvis, 1 L ankle, 2 R ankle, 3 L foot, 4 R foot
CONTACT_JOINTS = [1, 2, 3, 4]
FOOT_IDX = [2, 3]  # positions in CONTACT_JOINTS of the toe joints (planted on the ground)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def roty(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def exp_so3(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def make_walker(rng):
    """World y-up. Returns joints (L, 5, 3), contact (L, 4) bool."""
    # speed profile: stand, walk, stand, walk back toward camera, side-step, stand
    segs = [(1.5, 0.0, 0.0), (2.5, 0.9, 0.6), (1.0, 0.0, 0.0), (3.0, 0.8, -1.1), (2.0, 0.5, 1.6), (1.5, 0.0, 0.0)]  # room-sized
    speed, turn = [], []
    for dur, v, w in segs:
        n = int(dur * FPS)
        speed += [v] * n
        turn += [w] * n
    speed = gaussian_filter1d(np.array(speed), 6)
    turn = gaussian_filter1d(np.array(turn), 6)
    L = len(speed)
    heading = np.cumsum(turn) / FPS  # start walking across the view
    fwd = np.stack([np.cos(heading), np.zeros(L), np.sin(heading)], -1)
    side = np.stack([np.sin(heading), np.zeros(L), -np.cos(heading)], -1)
    root = np.cumsum(fwd * speed[:, None] / FPS, 0) + np.array([0.0, 0.0, 0.0])

    # footsteps: each foot plants for 0.6 of a 1.0 s stride when moving; both planted when still
    phase = np.cumsum(np.maximum(speed, 0) / 1.25) / FPS  # strides
    joints = np.zeros((L, 5, 3))
    contact = np.zeros((L, 4), bool)
    for f, sgn, off in ((0, 1, 0.0), (1, -1, 0.5)):
        ph = (phase + off) % 1.0
        moving = speed > 0.15
        planted = (ph < 0.6) | ~moving
        foot = np.zeros((L, 3))
        fdir = np.zeros((L, 3))  # direction the foot points; frozen while it is planted
        anchor = root[0] + side[0] * 0.1 * sgn
        anchor_dir = fwd[0]
        t = 0
        while t < L:
            if planted[t]:
                if t > 0 and not planted[t - 1]:  # new plant: land under the hip, slightly ahead
                    anchor = root[t] + side[t] * 0.1 * sgn + fwd[t] * 0.12 * (speed[t] > 0.15)
                    anchor[1] = 0.0
                    anchor_dir = fwd[t]
                foot[t], fdir[t] = anchor, anchor_dir
                t += 1
            else:
                s0 = t
                while t < L and not planted[t]:
                    t += 1
                s1 = min(t, L - 1)
                nxt = root[s1] + side[s1] * 0.1 * sgn + fwd[s1] * 0.12
                nxt[1] = 0.0
                for k in range(s0, t):
                    u = (k - s0 + 1) / (t - s0 + 1)
                    e = 0.5 - 0.5 * np.cos(np.pi * u)
                    foot[k] = anchor * (1 - e) + nxt * e
                    foot[k, 1] = 0.09 * np.sin(np.pi * u)
                    d = anchor_dir * (1 - e) + fwd[s1] * e
                    fdir[k] = d / np.linalg.norm(d)
        toe = foot + fdir * 0.13
        ankle = foot + np.array([0, 0.08, 0]) - fdir * 0.05
        joints[:, 1 + f] = ankle
        joints[:, 3 + f] = toe
        contact[:, f] = planted
        contact[:, 2 + f] = planted
    feet_mid = joints[:, 1:3].mean(1)
    pelvis = feet_mid.copy()
    pelvis[:, 1] = 0.93 + 0.02 * np.cos(4 * np.pi * phase)
    pelvis[:, [0, 2]] = gaussian_filter1d(feet_mid[:, [0, 2]], 4, axis=0)
    joints[:, 0] = pelvis
    return joints, contact


def predictions(rng, joints, contact, R_c2w_true, cam_pos):
    L = len(joints)
    R_w2c = R_c2w_true.T
    pel = joints[:, 0]
    pel_c = (pel - cam_pos) @ R_w2c.T
    s = 1.04
    smooth_noise = lambda shape, sig, k: gaussian_filter1d(rng.normal(0, 1, shape), k, axis=0) * sig * np.sqrt(2 * np.sqrt(np.pi) * k)
    # network outputs are temporally correlated: half white noise, half ~0.1 s correlated noise
    pel_c_pred = s * pel_c + (rng.normal(0, 1, (L, 3)) + smooth_noise((L, 3), 1.0, 3)) * np.array([0.015, 0.015, 0.08]) / np.sqrt(2)
    vel = np.diff(pel, axis=0, append=pel[-1:])
    vel_pred = (vel @ roty(np.radians(4)).T) * 0.93 + rng.normal(0, 0.004, (L, 3))
    offsets = joints - pel[:, None]
    offsets_pred = offsets + (rng.normal(0, 0.004, offsets.shape) + smooth_noise(offsets.shape, 0.01, 3))
    R_est = np.stack([R_c2w_true @ exp_so3(rng.normal(0, np.radians(1.5), 3)) for _ in range(L)])
    prob = gaussian_filter1d(contact.astype(float), 1.5, axis=0) * 0.9 + 0.05
    flip = rng.random(prob.shape) < 0.04
    prob[flip] = 1 - prob[flip]
    return pel_c_pred, vel_pred, offsets_pred, R_est, prob


def rollout(vel):
    return np.concatenate([np.zeros((1, 3)), np.cumsum(vel[:-1], 0)])


def drop_to_floor(p, offsets):
    p = p.copy()
    p[:, 1] -= (p[:, None, 1] + offsets[..., 1]).min()
    return p


def legacy(pel_c_pred, vel_pred, offsets_pred, R_est, prob):
    """Port of pp_static_joint_cam (pelvis + joint offsets instead of SMPL FK)."""
    L = len(vel_pred)
    p = rollout(vel_pred)
    c = gaussian_filter1d(pel_c_pred, 5, axis=0, mode="nearest")
    R_c2w = R_est[0]
    t_c2w = p[0] - R_c2w @ c[0]
    c_w = c @ R_c2w.T + t_c2w
    post = p.copy()
    thr = 0.25
    for i in range(1, L):
        d = post[i] - c_w[i]
        d = d * ~((d > -thr) & (d < thr))
        d = np.clip(d, -0.02, 0.02)
        post[i:] -= d
    j_static = post[:, None] + offsets_pred[:, CONTACT_JOINTS]
    disp = j_static[1:] - j_static[:-1]
    lab = prob[:-1] > 0.8
    n = np.maximum(lab.sum(-1, keepdims=True), 1)
    pd = (disp * lab[..., None]).sum(1) / n
    pd[:, 1] = 0
    for i in range(1, L):
        post[i:] -= pd[i - 1]
    return drop_to_floor(post, offsets_pred)


def lsq(pel_c_pred, vel_pred, offsets_pred, R_est, prob, **kw):
    R_c2w = scs.chordal_mean(R_est)
    p, info = scs.solve_static_cam_root(
        pel_c_pred, offsets_pred, vel_pred, prob, CONTACT_JOINTS, R_c2w, foot_joints=FOOT_IDX, **kw
    )
    return p


def align_xz(est, true):
    """Best yaw + translation in the ground plane (heading and origin are unobservable)."""
    e, t = est[:, [0, 2]], true[:, [0, 2]]
    em, tm = e.mean(0), t.mean(0)
    U, _, Vt = np.linalg.svd((e - em).T @ (t - tm))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    out = est.copy()
    out[:, [0, 2]] = (e - em) @ R.T + tm
    return out


def legacy_feet(p, offsets_pred, prob):
    """Port of process_ik's target blending: target_t = target_{t-1} * c + pos_t * (1 - c)."""
    pos = p[:, None] + offsets_pred[:, CONTACT_JOINTS]
    tgt = pos.copy()
    for t in range(1, len(pos)):
        c = prob[t - 1][:, None]
        tgt[t] = tgt[t - 1] * c + pos[t] * (1 - c)
    return tgt


def lock_feet(p, offsets_pred, prob):
    pos = p[:, None] + offsets_pred[:, CONTACT_JOINTS]
    tgt, w = contact_lock_targets(pos, prob, ground=FOOT_IDX)
    return tgt


def metrics(p_est, offsets_pred, joints, contact, feet=None):
    true_p = joints[:, 0]
    p = align_xz(p_est, true_p)
    if feet is None:
        feet = p[:, None] + offsets_pred[:, CONTACT_JOINTS]
    else:  # feet given in the estimate's frame: apply the same alignment
        moved = p - p_est
        e, t = p_est[:, [0, 2]], true_p[:, [0, 2]]
        em, tm = e.mean(0), t.mean(0)
        U, _, Vt = np.linalg.svd((e - em).T @ (t - tm))
        R = (U @ Vt).T
        f = feet.copy()
        f[..., [0, 2]] = (feet[..., [0, 2]] - em) @ R.T + tm
        feet = f
    st = contact[:-1] & contact[1:]
    horiz = np.linalg.norm((feet[1:] - feet[:-1])[..., [0, 2]], axis=-1)
    toe_y = feet[:, FOOT_IDX, 1][contact[:, FOOT_IDX]]  # true value is 0
    acc = np.linalg.norm(p[2:] - 2 * p[1:-1] + p[:-2], axis=-1)
    # sliding: how far each planted foot travels between touchdown and lift-off
    slides = []
    for ci in range(contact.shape[1]):
        c = contact[:, ci]
        starts = np.nonzero(c & ~np.r_[False, c[:-1]])[0]
        ends = np.nonzero(c & ~np.r_[c[1:], False])[0]
        for a, b in zip(starts, ends):
            if b - a >= 3:
                slides.append(np.linalg.norm(feet[b, ci, [0, 2]] - feet[a, ci, [0, 2]]))
    return {
        "root_err_cm": float(np.linalg.norm(p - true_p, axis=-1).mean() * 100),
        "root_err_end_cm": float(np.linalg.norm(p[-1] - true_p[-1]) * 100),
        "foot_skate_mm": float(horiz[st].mean() * 1000),
        "slide_cm": float(np.mean(slides) * 100),
        "contact_height_cm": float(np.abs(toe_y).mean() * 100),
        "below_floor_pct": float((toe_y < -0.01).mean() * 100),
        "accel_mm": float(acc.mean() * 1000),
        "path": p[::2, [0, 2]].round(3).tolist(),
        "height": p[::2, 1].round(3).tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed0", type=int, default=0)
    args = ap.parse_args()
    # webcam-ish: 1.3 m high, 3.5 m back from the start, tilted 12 deg down, yaw 20 deg
    j0, _ = make_walker(np.random.default_rng(0))
    centre = (j0[:, 0].min(0) + j0[:, 0].max(0)) / 2
    cam_pos = centre + np.array([0.6, 0.0, 0.0])
    cam_pos[1] = 1.3
    cam_pos[2] = j0[:, 0, 2].min() - 2.6  # 2.6 m in front of the nearest point
    zc = centre - cam_pos
    zc[1] = (0.7 - cam_pos[1])  # aim at chest height, so the camera tilts down
    zc /= np.linalg.norm(zc)
    xc = np.cross(np.array([0, -1.0, 0]), zc); xc /= np.linalg.norm(xc)  # image-down is world-down
    yc = np.cross(zc, xc)
    R_c2w = np.stack([xc, yc, zc], -1)
    keys = ["root_err_cm", "root_err_end_cm", "foot_skate_mm", "slide_cm", "contact_height_cm", "below_floor_pct", "accel_mm"]
    pc0 = (j0[:, 0] - cam_pos) @ R_c2w
    print(f"camera view: depth {pc0[:, 2].min():.1f}-{pc0[:, 2].max():.1f} m, |x/z| max {np.abs(pc0[:, 0] / pc0[:, 2]).max():.2f}, "
          f"path {np.ptp(j0[:, 0, 0]):.1f} x {np.ptp(j0[:, 0, 2]):.1f} m, {len(j0) / FPS:.1f} s")
    methods = ("raw", "legacy", "legacy+ik", "lsq", "lsq+lock")
    agg = {m: {k: [] for k in keys} for m in methods}
    keep = None
    for seed in range(args.seed0, args.seed0 + args.seeds):
        rng = np.random.default_rng(seed)
        joints, contact = make_walker(rng)
        pc, v, o, Rs, prob = predictions(rng, joints, contact, R_c2w, cam_pos)
        res = {
            "raw": drop_to_floor(rollout(v), o),
            "legacy": legacy(pc, v, o, Rs, prob),
            "lsq": lsq(pc, v, o, Rs, prob),
        }
        ms = {m: metrics(p, o, joints, contact) for m, p in res.items()}
        ms["legacy+ik"] = metrics(res["legacy"], o, joints, contact, legacy_feet(res["legacy"], o, prob))
        ms["lsq+lock"] = metrics(res["lsq"], o, joints, contact, lock_feet(res["lsq"], o, prob))
        ms = {m: ms[m] for m in methods}
        for m in ms:
            for k in keys:
                agg[m][k].append(ms[m][k])
        if seed == args.seed0:
            keep = {"truth": joints[::2, 0][:, [0, 2]].round(3).tolist(), "truth_height": joints[::2, 0, 1].round(3).tolist(),
                    **{m: {"path": ms[m]["path"], "height": ms[m]["height"]} for m in ms}}
    print(f"{'':10s}" + "".join(f"{k:>18s}" for k in keys))
    summary = {}
    for m in agg:
        summary[m] = {k: float(np.mean(agg[m][k])) for k in keys}
        print(f"{m:10s}" + "".join(f"{summary[m][k]:18.2f}" for k in keys))
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "example": keep, "fps": FPS, "every": 2}))


if __name__ == "__main__":
    main()
