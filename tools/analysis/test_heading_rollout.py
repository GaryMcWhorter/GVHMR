"""
End-to-end check of the heading rollout on synthetic ground truth.

Builds a camera path and a walking/turning subject in a z-up world, derives the
exact inputs GVHMR's network is trained to predict (global_orient_c,
global_orient_gv from get_R_c2gv, cam_angvel from compute_cam_angvel), then
runs the rollouts and measures how far the recovered world heading strays from
the truth. A perfect network would therefore score ~0 here; anything left over
is error introduced by the rollout itself.

Usage (no SMPL, GPU, or checkpoints needed):
    python tools/analysis/test_heading_rollout.py
Heavy optional deps (smplx, wis3d, pytorch3d.ops, ...) are stubbed if missing.
"""
import importlib.abc, importlib.machinery, sys, types
from unittest import mock

class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    STUB = ("smplx", "wis3d", "pytorch3d.ops", "pytorch3d.renderer", "pytorch3d.structures",
            "cv2", "imageio", "open3d", "tkinter", "turtle")
    def find_spec(self, name, path, target=None):
        if any(name == s or name.startswith(s + ".") for s in self.STUB):
            try:
                sys.meta_path.remove(self); found = importlib.util.find_spec(name)
            except (ImportError, ValueError):
                found = None
            finally:
                sys.meta_path.insert(0, self)
            if found is None:
                return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None
    def create_module(self, spec):
        m = mock.MagicMock(name=spec.name); m.__path__ = []; m.__spec__ = spec; return m
    def exec_module(self, module): pass
import importlib.util
sys.meta_path.insert(0, _StubFinder())

import math
import torch
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from hmr4d.utils.geo.hmr_global import get_R_c2gv
from hmr4d.utils.geo_transform import compute_cam_angvel
from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import get_smpl_params_w_Rt_v2, get_smpl_params_w_Rt_v3

d = math.radians

def rot(axis, ang):
    return axis_angle_to_matrix(torch.tensor(axis, dtype=torch.float64) * ang)

def make_case(L, pitch, roll, pan, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0., -1, 0], [0, 0, -1], [1, 0, 0]], dtype=torch.float64)  # level cam looking +x, image-down = -z
    R_w2c, R_body = [], []
    for t in range(L):
        R_c2w = rot([0, 0, 1], pan(t)) @ base.T @ rot([1, 0, 0], -pitch(t, g)) @ rot([0, 0, 1], roll(t, g))
        R_w2c.append(R_c2w.T)
        # body: SMPL-style y-up stand, turning about world z at 40 deg/s with sway
        stand = torch.tensor([[1., 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64).T
        R_body.append(rot([0, 0, 1], d(40) * t / 30) @ stand @ rot([0, 0, 1], d(3) * math.sin(t / 6)))
    R_w2c, R_body = torch.stack(R_w2c), torch.stack(R_body)
    g_w = [0, 0, -1]
    R_c = R_w2c @ R_body
    return dict(
        R_body=R_body,
        global_orient_c=matrix_to_axis_angle(R_c)[None].float(),
        global_orient_gv={m: matrix_to_axis_angle(get_R_c2gv(R_w2c.float(), g_w, mode=m).double() @ R_c)[None].float()
                          for m in ("view", "rotor")},
        cam_angvel=compute_cam_angvel(R_w2c)[None].float(),
        local_transl_vel=torch.zeros(1, L, 3).float(),
    )

def heading_err_deg(out_aa, R_body):
    """out = W R_body for a constant W if perfect; report max drift of W from W_0 (degrees)."""
    R_out = axis_angle_to_matrix(out_aa[0].double())
    W = R_out @ R_body.mT
    rel = W @ W[:1].mT
    cos = ((rel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1)
    e = torch.rad2deg(torch.acos(cos))
    return e.max().item(), e.mean().item()

def run(name, case_kwargs):
    c = make_case(**case_kwargs)
    common = dict(local_transl_vel=c["local_transl_vel"], global_orient_c=c["global_orient_c"],
                  cam_angvel=c["cam_angvel"])
    rows = [
        ("v2 (original)",       get_smpl_params_w_Rt_v2(global_orient_gv=c["global_orient_gv"]["view"], **common)),
        ("v3 view, no smooth",  get_smpl_params_w_Rt_v3(global_orient_gv=c["global_orient_gv"]["view"],  gv_frame="view",  smooth=False, **common)),
        ("v3 rotor, no smooth", get_smpl_params_w_Rt_v3(global_orient_gv=c["global_orient_gv"]["rotor"], gv_frame="rotor", smooth=False, **common)),
        ("v3 rotor, smoothed",  get_smpl_params_w_Rt_v3(global_orient_gv=c["global_orient_gv"]["rotor"], gv_frame="rotor", **common)),
    ]
    print(f"\n{name}")
    for label, out in rows:
        mx, mn = heading_err_deg(out["global_orient"], c["R_body"])
        print(f"  {label:22s} heading error  max {mx:8.3f}  mean {mn:8.3f} deg")

if __name__ == "__main__":
    L = 300
    zero = lambda t, g: 0.0
    run("1. Handheld: 10 deg pitch, no roll, panning 6 deg/s",
        dict(L=L, pitch=lambda t, g: d(10), roll=zero, pan=lambda t: d(0.2) * t))
    run("2. Walking cameraman: 15 deg pitch +/-3, roll +/-4, panning",
        dict(L=L, pitch=lambda t, g: d(15 + 3 * math.sin(t / 7)), roll=lambda t, g: d(4 * math.sin(t / 5)),
             pan=lambda t: d(0.2) * t))
    run("3. Drone tilting 80 -> ~90 deg down, +/-1 deg roll",
        dict(L=L, pitch=lambda t, g: d(80 + 9.9 * t / L + 0.3 * math.sin(t / 3)),
             roll=lambda t, g: d(math.sin(t / 4)), pan=lambda t: d(0.05) * t))
    run("4. Near-nadir mount, 0.2 deg jitter",
        dict(L=L, pitch=lambda t, g: d(89.7 + 0.2 * torch.randn(1, generator=g).item()),
             roll=lambda t, g: d(0.2 * torch.randn(1, generator=g).item()), pan=lambda t: 0.0))
