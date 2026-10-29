"""
process_footlock on a toy SMPL-shaped skeleton (SMPL's 22-joint tree, approximate rest pose):
a planted foot must stay put and sit on the floor; nothing outside the legs may change.

    PYTHONPATH=<pytorch3d shim>:. python tools/analysis/test_process_footlock.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import test_heading_rollout  # noqa: F401  (stubs for heavy optional imports)

import torch
from pytorch3d.transforms import axis_angle_to_matrix
import hmr4d.utils.matrix as matrix
from hmr4d.model.gvhmr.utils.postprocess import process_footlock

PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
REST = torch.tensor([
    [0, .93, 0], [.07, .84, 0], [-.07, .84, 0], [0, 1.03, 0], [.1, .45, 0], [-.1, .45, 0], [0, 1.17, 0],
    [.1, .05, -.03], [-.1, .05, -.03], [0, 1.22, 0], [.11, 0, .1], [-.11, 0, .1], [0, 1.45, 0], [.08, 1.37, 0],
    [-.08, 1.37, 0], [0, 1.55, 0], [.18, 1.4, 0], [-.18, 1.4, 0], [.45, 1.4, 0], [-.45, 1.4, 0],
    [.7, 1.4, 0], [-.7, 1.4, 0]])


class ToyEndecoder:
    parents = PARENTS

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        B, L = body_pose.shape[:2]
        aa = torch.cat([global_orient, body_pose], -1).reshape(B, L, -1, 3)
        R = axis_angle_to_matrix(aa)
        local = REST - REST[PARENTS].where(torch.tensor(PARENTS)[:, None] >= 0, torch.zeros(3))
        local = local.expand(B, L, 22, 3).clone()
        if transl is not None:
            local[..., 0, :] += transl
        mat = matrix.get_TRS(R, local)
        fk = matrix.forward_kinematics(mat, PARENTS)
        j = matrix.get_position(fk)
        return (j, mat, fk) if get_intermediate else j


def main():
    torch.manual_seed(0)
    L = 60
    t = torch.arange(L).float()
    body_pose = (torch.randn(1, L, 21, 3) * 0.04 + 0.03 * torch.sin(t / 5)[None, :, None, None]).reshape(1, L, 63)
    body_pose[:, :, 3 * 3 + 0] += 0.4  # left knee (joint 4 -> body_pose index 3) bent
    body_pose[:, :, 4 * 3 + 0] += 0.4
    go = torch.zeros(1, L, 3)
    go[..., 1] = 0.01 * t
    transl = torch.stack([0.004 * t, 0.02 * torch.ones(L), torch.zeros(L)], -1)[None]
    logits = torch.full((1, L, 6), -6.0)
    logits[:, 15:40, [0, 1]] = 6.0  # left ankle + toe planted frames 15..39
    outputs = {"pred_smpl_params_global": {"body_pose": body_pose, "betas": torch.zeros(1, L, 10),
                                           "global_orient": go, "transl": transl},
               "static_conf_logits": logits}
    enc = ToyEndecoder()
    j_before = enc.fk_v2(**outputs["pred_smpl_params_global"])
    new_pose = process_footlock(outputs, enc, pin_to_floor=True)
    j_after = enc.fk_v2(**{**outputs["pred_smpl_params_global"], "body_pose": new_pose})
    lt = j_after[0, 15:40, 10]
    la = j_after[0, 15:40, 7]
    ok = True
    still = max((lt - lt.mean(0)).abs().max().item(), (la - la.mean(0)).abs().max().item())
    floor = lt[:, 1].abs().max().item()
    moved_before = (j_before[0, 15:40, 10] - j_before[0, 15, 10]).norm(dim=-1).max().item()
    good = still < 1e-3 and floor < 1e-3
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  planted left foot: moves {still * 1000:.2f} mm (was {moved_before * 100:.1f} cm), "
          f"toe height {floor * 1000:.2f} mm")
    legs = {1, 2, 4, 5, 7, 8, 10, 11}
    others = [j for j in range(22) if j not in legs]
    d = (j_after[0][:, others] - j_before[0][:, others]).abs().max().item()
    free = (j_after[0, :5, 10] - j_before[0, :5, 10]).abs().max().item()
    good = d < 1e-4 and free < 1e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  non-leg joints unchanged ({d:.1e}); foot untouched far from contact ({free:.1e})")
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
