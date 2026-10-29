"""
Checks for hmr4d/model/gvhmr/utils/foot_lock.py on a toy leg (no SMPL files needed).

    PYTHONPATH=. python tools/analysis/test_foot_lock.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hmr4d.model.gvhmr.utils.foot_lock import contact_intervals, contact_lock_targets, two_bone_ik  # noqa: E402
import hmr4d.utils.matrix as matrix  # noqa: E402

PARENTS = [-1, 0, 1, 2, 3]  # root, hip, knee, ankle, toe
OFFS = torch.tensor([[0, 0.95, 0], [0.1, -0.05, 0], [0, -0.42, 0], [0, -0.42, 0], [0, -0.06, 0.14]])


def rotx(a):
    c, s = torch.cos(a), torch.sin(a)
    o, z = torch.ones_like(a), torch.zeros_like(a)
    return torch.stack([o, z, z, z, c, -s, z, s, c], -1).reshape(a.shape + (3, 3))


def leg(hip_flex, knee_flex):
    """(N,) angles -> local mats (1, N, 5, 4, 4); knee bends backward like a human knee."""
    N = hip_flex.shape[0]
    R = torch.eye(3).repeat(N, 5, 1, 1)
    R[:, 1] = rotx(-hip_flex)
    R[:, 2] = rotx(knee_flex)
    R[:, 3] = rotx(-0.2 * torch.ones(N))
    M = torch.eye(4).repeat(N, 5, 1, 1)
    M[..., :3, :3] = R
    M[..., :3, 3] = OFFS
    return M[None]


def main():
    ok = True
    g = torch.Generator().manual_seed(0)
    N = 200
    local = leg(torch.rand(N, generator=g) * 0.8 - 0.2, torch.rand(N, generator=g) * 1.2 + 0.15)
    glob = matrix.forward_kinematics(local, PARENTS)
    p_a = glob[..., 3, :3, 3]
    # reachable targets near the current ankle
    tgt = p_a + (torch.rand(1, N, 3, generator=g) - 0.5) * 0.25
    out = two_bone_ik(local, PARENTS, 1, 2, 3, 4, tgt, torch.ones(1, N))
    g2 = matrix.forward_kinematics(out, PARENTS)
    reach = (g2[..., 3, :3, 3] - tgt).norm(dim=-1)
    reachable = ((tgt - glob[..., 1, :3, 3]).norm(dim=-1) < 0.83)
    def bend_normal(G):
        h, k, a = G[..., 1, :3, 3], G[..., 2, :3, 3], G[..., 3, :3, 3]
        return torch.cross(k - h, a - k, dim=-1)
    # the knee keeps bending the same way (bend-plane normal keeps its direction): no flips
    knee_forward = ((bend_normal(g2) * bend_normal(glob)).sum(-1) > 0)[reachable].float().mean().item()
    ank_rot = (g2[..., 3, :3, :3] - glob[..., 3, :3, :3]).abs().max().item()
    good = reach[reachable].max().item() < 1e-4 and knee_forward > 0.99 and ank_rot < 1e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  ankle reaches target: max err {reach[reachable].max().item() * 1000:.3f} mm "
          f"over {int(reachable.sum())} reachable; knee kept forward {knee_forward:.0%}; ankle orientation kept {ank_rot:.1e}")

    out0 = two_bone_ik(local, PARENTS, 1, 2, 3, 4, tgt, torch.zeros(1, N))
    d0 = (out0 - local).abs().max().item()
    ok &= d0 < 1e-5
    print(f"{'PASS' if d0 < 1e-5 else 'FAIL'}  zero weight leaves the leg unchanged ({d0:.1e})")

    toe_t = g2[..., 4, :3, 3] + torch.tensor([0.0, -0.03, 0.02])
    out_t = two_bone_ik(local, PARENTS, 1, 2, 3, 4, tgt, torch.ones(1, N), toe_t, torch.ones(1, N))
    g3 = matrix.forward_kinematics(out_t, PARENTS)
    v1 = g3[..., 4, :3, 3] - g3[..., 3, :3, 3]
    v2 = toe_t - g3[..., 3, :3, 3]
    ang = torch.rad2deg(torch.acos((torch.nn.functional.normalize(v1, dim=-1) * torch.nn.functional.normalize(v2, dim=-1)).sum(-1).clamp(-1, 1)))
    good = ang.max().item() < 0.05 and (g3[..., 3, :3, 3] - tgt).norm(dim=-1)[reachable].max().item() < 1e-4
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  toe points at its target: max {ang.max().item():.3f} deg, ankle still on target")

    # lock targets: hysteresis ignores a 1-frame dropout, lock sits on the floor
    prob = np.r_[np.zeros(5), np.full(10, 0.9), [0.3], np.full(10, 0.9), np.zeros(5)]
    iv = contact_intervals(prob)
    good = iv == [(5, 14), (16, 25)] or iv == [(5, 25)]
    pos = np.zeros((len(prob), 1, 3)) + np.linspace(0, 0.05, len(prob))[:, None, None]
    tgt_l, w = contact_lock_targets(pos, prob[:, None], ground=(0,))
    still = np.abs(np.diff(tgt_l[5:15, 0], axis=0)).max()
    good &= still < 1e-12 and np.allclose(tgt_l[5:15, 0, 1], 0) and w[0, 0] == 0 and 0 < w[3, 0] < 1
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  lock intervals {iv}, locked target still, on the floor, eased edges")
    print("OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
