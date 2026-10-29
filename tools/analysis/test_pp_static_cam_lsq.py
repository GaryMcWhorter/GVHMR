"""
Convention check for pp_static_cam_lsq: with exact, noise-free inputs from a static camera, the
recovered global translation must match the truth (up to the floor shift). Uses a toy rigid
skeleton in place of SMPL, so no body-model files are needed.

    PYTHONPATH=<pytorch3d shim>:. python tools/analysis/test_pp_static_cam_lsq.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import test_heading_rollout  # noqa: F401  (installs stubs for heavy optional imports)

import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.model.gvhmr.utils.postprocess import pp_static_cam_lsq

torch.manual_seed(0)
REST = torch.randn(22, 3).double() * 0.3
REST[0] = torch.tensor([0.0, 0.9, 0.02])  # pelvis rest position is not at the origin, like SMPL
REST[[10, 11]] = torch.tensor([[0.1, 0.0, 0.1], [-0.1, 0.0, 0.1]]).double()  # toes on the floor


class ToyEndecoder:
    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        R = axis_angle_to_matrix(global_orient.double())
        j = torch.einsum("blij,kj->blki", R, REST - REST[0]) + REST[0]
        if transl is not None:
            j = j + transl.double()[:, :, None]
        return j.to(body_pose.dtype)


def main():
    L = 120
    t = torch.arange(L).double()
    yaw = 0.02 * t
    go_w = torch.stack([torch.zeros(L), yaw, torch.zeros(L)], -1).double()  # y-up turning
    transl_w = torch.stack([0.01 * t, torch.zeros(L), 0.5 * torch.sin(t / 30)], -1).double()
    # static camera 3 m away looking at the walker, tilted down 10 deg
    R_c2w = axis_angle_to_matrix(torch.tensor([0.0, 3.14159, 0.0]).double()) @ axis_angle_to_matrix(
        torch.tensor([3.14159 - 0.17, 0, 0]).double())
    cam = torch.tensor([0.5, 1.4, 3.0]).double()
    R_w2c = R_c2w.T
    go_c = matrix_to_axis_angle(R_w2c @ axis_angle_to_matrix(go_w))
    pel_w = transl_w + REST[0]
    pel_c = (pel_w - cam) @ R_w2c.T
    transl_c = pel_c - (R_w2c @ REST[0]) * 0 - REST[0]  # incam fk: pelvis_c = transl_c + REST[0]
    z = torch.zeros(1, L, 63).double()
    b = torch.zeros(1, L, 10).double()
    outputs = {
        "pred_smpl_params_incam": {"body_pose": z, "betas": b, "global_orient": go_c[None], "transl": transl_c[None]},
        "pred_smpl_params_global": {"body_pose": z, "betas": b, "global_orient": go_w[None],
                                    "transl": transl_w[None].clone()},
        "static_conf_logits": torch.full((1, L, 6), -5.0).double(),  # no contacts: camera term must carry it
    }
    def dev(out):  # compare after removing the floor shift and the gauge
        d = out - transl_w
        return (d - d.mean(0)).abs().max().item()

    ok = True
    e = dev(pp_static_cam_lsq(outputs, ToyEndecoder())[0])
    ok &= e < 3e-3
    print(f"{"PASS" if e < 3e-3 else "FAIL"}  exact inputs: max deviation {e * 1000:.2f} mm (smoothness prior bias only)")

    drift = 0.3 * torch.tensor([1.0, 0, 0.5]).double() * (t[:, None] / L)
    outputs["pred_smpl_params_global"]["transl"] = (transl_w + drift)[None]
    e = dev(pp_static_cam_lsq(outputs, ToyEndecoder())[0])
    e_in = (drift - drift.mean(0)).abs().max().item()
    ok &= e < 0.5 * e_in
    print(f"{"PASS" if e < 0.5 * e_in else "FAIL"}  rollout drifting {e_in * 100:.0f} cm: max deviation {e * 100:.1f} cm")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
