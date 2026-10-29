"""
Top-down walking paths recovered by the heading rollout, original (v2) vs rotor frame (v3),
from perfect network outputs. A subject walks a curving path; the camera is either a
handheld (both should match truth) or a near-overhead mount (v2's GV frame is singular).

    python tools/analysis/gv_rollout_paths.py --out gv_paths.json
"""

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import test_heading_rollout as T  # installs the import stubs  # noqa: E402

import torch  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402
from hmr4d.utils.geo.hmr_global import get_R_c2gv, get_local_transl_vel  # noqa: E402
from hmr4d.utils.geo_transform import compute_cam_angvel  # noqa: E402
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2, get_smpl_params_w_Rt_v3  # noqa: E402

d = math.radians


def scenario(L, pitch, roll, pan, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float64)
    stand = torch.tensor([[1.0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64).T  # SMPL y-up -> world z-up
    R_w2c, R_body, pos = [], [], [torch.zeros(3, dtype=torch.float64)]
    for t in range(L):
        R_c2w = T.rot([0, 0, 1], pan(t)) @ base.T @ T.rot([1, 0, 0], -pitch(t, g)) @ T.rot([0, 0, 1], roll(t, g))
        R_w2c.append(R_c2w.T)
        # walk: turn rate varies (S-curve), slight sway
        yaw = d(70) * math.sin(t / 90) + d(25) * t / 30
        Rb = T.rot([0, 0, 1], yaw) @ stand @ T.rot([0, 1, 0], d(4) * math.sin(t / 5))
        R_body.append(Rb)
        if t:
            fwd = R_body[-2] @ torch.tensor([0.0, 0, 1], dtype=torch.float64)
            fwd[2] = 0
            pos.append(pos[-1] + fwd / fwd.norm() * 1.3 / 30)
    R_w2c, R_body, pos = torch.stack(R_w2c), torch.stack(R_body), torch.stack(pos)
    R_c = R_w2c @ R_body
    go_w = matrix_to_axis_angle(R_body)
    common = dict(
        global_orient_c=matrix_to_axis_angle(R_c)[None].float(),
        cam_angvel=compute_cam_angvel(R_w2c)[None].float(),
        local_transl_vel=get_local_transl_vel(pos[None], go_w[None]).float(),
    )
    gv = {m: matrix_to_axis_angle(get_R_c2gv(R_w2c.float(), [0, 0, -1], mode=m).double() @ R_c)[None].float()
          for m in ("view", "rotor")}
    return R_body, pos, common, gv


def to_truth_frame(out, R_body):
    """Rotate the estimate's world into the truth world using frame 0 only."""
    R_out = axis_angle_to_matrix(out["global_orient"][0].double())
    W0 = R_out[0] @ R_body[0].T
    p = (W0.T @ out["transl"][0].double().T).T
    return p - p[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="gv_paths.json")
    args = ap.parse_args()
    L = 450
    zero = lambda t, g: 0.0
    cases = {
        "handheld": dict(pitch=lambda t, g: d(15 + 3 * math.sin(t / 7)), roll=lambda t, g: d(4 * math.sin(t / 5)),
                         pan=lambda t: d(0.25) * t),
        "overhead": dict(pitch=lambda t, g: d(89.7 + 0.2 * torch.randn(1, generator=g).item()),
                         roll=lambda t, g: d(0.2 * torch.randn(1, generator=g).item()), pan=lambda t: 0.0),
    }
    res = {}
    for name, kw in cases.items():
        R_body, pos, common, gv = scenario(L, **kw)
        v2 = get_smpl_params_w_Rt_v2(global_orient_gv=gv["view"], **common)
        v3 = get_smpl_params_w_Rt_v3(global_orient_gv=gv["rotor"], gv_frame="rotor", **common)
        res[name] = {"truth": pos[:, :2].tolist()}
        for key, out in (("v2", v2), ("rotor", v3)):
            p = to_truth_frame(out, R_body)
            herr, _ = T.heading_err_deg(out["global_orient"], R_body)
            res[name][key] = p[:, :2].tolist()
            res[name][key + "_end_err_m"] = float((p[-1, :2] - pos[-1, :2]).norm())
            res[name][key + "_heading_max_deg"] = herr
            print(f"{name:9s} {key:6s} end error {res[name][key + '_end_err_m']:.3f} m  max heading err {herr:.2f} deg")
    Path(args.out).write_text(json.dumps(res))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
