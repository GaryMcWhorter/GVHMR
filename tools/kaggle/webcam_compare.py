"""
Real-footage comparison for static (webcam) cameras: legacy vs lsq post-processing.

Runs the network ONCE per video (postproc off), then applies both post-processing paths to
copies of the same raw outputs, so every difference comes from post-processing alone.

There is no ground truth on real footage, so the metrics use the model's own contact
predictions as the reference for "this foot should be planted":
  slide_cm          horizontal travel of a planted toe from touchdown to lift-off
  skate_mm          per-frame horizontal motion of planted toes
  toe_height_cm     planted toe height above the floor each method uses
  toe_height_sd_cm  spread of planted toe heights (floor-independent: one floor or many?)
  cam_gap_cm        distance between the world pelvis and where the static camera sees it
                    (world path re-projected through the averaged camera pose)
  pelvis_jerk_mm    mean |second difference| of the pelvis per frame
  pose_jitter_deg   mean frame-to-frame change of joint rotations, minus a smoothed version

Usage (from the repo root, after setup; see tools/kaggle/README.md):
    python tools/kaggle/webcam_compare.py --video docs/example_video/tennis.mp4 [--video my_webcam.mp4]
        [--out outputs/webcam_compare] [--render]
"""

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tools" / "demo"))

import hydra  # noqa: E402
from hydra import compose, initialize_config_module  # noqa: E402
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle  # noqa: E402

from hmr4d.configs import register_store_gvhmr  # noqa: E402
from hmr4d.utils.geo.hmr_cam import normalize_kp2d  # noqa: E402
from hmr4d.utils.net_utils import detach_to_cpu  # noqa: E402
from hmr4d.utils.video_io_utils import get_video_lwh, get_video_reader, get_writer  # noqa: E402
from hmr4d.model.gvhmr.utils.postprocess import (  # noqa: E402
    pp_static_joint_cam,
    process_ik,
    pp_static_cam_lsq,
    process_footlock,
    smooth_pose_rotations,
)
from hmr4d.model.gvhmr.utils.foot_lock import contact_intervals  # noqa: E402
from hmr4d.model.gvhmr.utils.static_cam_solve import chordal_mean  # noqa: E402
import demo as demo_mod  # noqa: E402  (tools/demo/demo.py: preprocessing and rendering)

TOES = [10, 11]  # L_foot, R_foot
TOE_PROB_IDX = [1, 3]  # their columns in static_conf_logits


def make_cfg(video, out_root):
    length, w, h = get_video_lwh(video)
    with initialize_config_module(version_base="1.3", config_module="hmr4d.configs"):
        register_store_gvhmr()
        cfg = compose(config_name="demo", overrides=[
            f"video_name={Path(video).stem}", "static_cam=True", f"output_root={out_root}", "use_dpvo=False"])
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)
    if not Path(cfg.video_path).exists():
        reader, writer = get_video_reader(video), get_writer(cfg.video_path, fps=30, crf=23)
        for img in reader:
            writer.write_frame(img)
        writer.close()
        reader.close()
    return cfg


def apply_pp(raw, enc, mode):
    o = copy.deepcopy(raw)
    if mode == "lsq":
        smooth_pose_rotations(o)
        o["pred_smpl_params_global"]["transl"] = pp_static_cam_lsq(o, enc)
        body_pose = process_footlock(o, enc, pin_to_floor=True)
    else:
        o["pred_smpl_params_global"]["transl"] = pp_static_joint_cam(o, enc)
        body_pose = process_ik(o, enc)
    o["pred_smpl_params_global"]["body_pose"] = body_pose
    o["pred_smpl_params_incam"]["body_pose"] = body_pose
    return o


@torch.no_grad()
def metrics(o, enc):
    g = o["pred_smpl_params_global"]
    j = enc.fk_v2(**g)[0].double().cpu().numpy()  # (L, 22, 3)
    prob = o["static_conf_logits"].sigmoid()[0].double().cpu().numpy()
    slides, skates, heights = [], [], []
    for jt, pc in zip(TOES, TOE_PROB_IDX):
        for s, e in contact_intervals(prob[:, pc], on=0.6, off=0.4, min_len=3):
            p = j[s:e + 1, jt]
            slides.append(np.linalg.norm(p[-1, [0, 2]] - p[0, [0, 2]]))
            skates.extend(np.linalg.norm(np.diff(p[:, [0, 2]], axis=0), axis=-1))
            heights.extend(p[:, 1])
    heights = np.array(heights)
    # both methods put their floor at y = 0 (lsq: fitted to planted toes; legacy: lowest joint)
    floor = 0.0
    # camera consistency: world pelvis vs static camera's view of the pelvis
    inc = o["pred_smpl_params_incam"]
    pc = enc.fk_v2(**inc)[0, :, 0].double().cpu().numpy()
    Rw = axis_angle_to_matrix(g["global_orient"][0]).double()
    Rc = axis_angle_to_matrix(inc["global_orient"][0]).double()
    R_c2w = chordal_mean((Rw @ Rc.mT).cpu().numpy())
    pw = j[:, 0]
    pc_w = pc @ R_c2w.T
    t = (pw - pc_w).mean(0)
    cam_gap = np.linalg.norm(pw - (pc_w + t), axis=-1)
    jerk = np.linalg.norm(pw[2:] - 2 * pw[1:-1] + pw[:-2], axis=-1)
    aa = g["body_pose"][0].reshape(-1, 21, 3)
    R = axis_angle_to_matrix(aa).double()
    step = torch.rad2deg(matrix_to_axis_angle(R[1:] @ R[:-1].mT).norm(dim=-1)).cpu().numpy()  # (L-1, 21)
    k = np.ones(5) / 5
    smooth = np.stack([np.convolve(step[:, i], k, mode="same") for i in range(21)], 1)
    return {
        "frames": int(len(j)),
        "planted_intervals": len(slides),
        "slide_cm": float(np.mean(slides) * 100) if slides else None,
        "skate_mm": float(np.mean(skates) * 1000) if skates else None,
        "toe_height_cm": float(np.mean(heights - floor) * 100) if len(heights) else None,
        "toe_height_sd_cm": float(np.std(heights) * 100) if len(heights) else None,
        "cam_gap_cm": float(cam_gap.mean() * 100),
        "pelvis_jerk_mm": float(jerk.mean() * 1000),
        "pose_jitter_deg": float(np.abs(step - smooth).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", action="append", required=True)
    ap.add_argument("--out", default="outputs/webcam_compare")
    ap.add_argument("--render", action="store_true", help="render global views for both methods (slow)")
    args = ap.parse_args()

    report = {}
    model = None
    for video in args.video:
        cfg = make_cfg(video, args.out)
        demo_mod.run_preprocess(cfg)
        data = demo_mod.load_data_dict(cfg)
        if model is None:
            model = hydra.utils.instantiate(cfg.model, _recursive_=False)
            model.load_pretrained_model(cfg.ckpt_path)
            model = model.eval().cuda()
        batch = {
            "length": data["length"][None],
            "obs": normalize_kp2d(data["kp2d"], data["bbx_xys"])[None],
            "bbx_xys": data["bbx_xys"][None],
            "K_fullimg": data["K_fullimg"][None],
            "cam_angvel": data["cam_angvel"][None],
            "f_imgseq": data["f_imgseq"][None],
        }
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            raw = model.pipeline.forward(batch, train=False, postproc=False, static_cam=True)
        enc = model.pipeline.endecoder
        report[video] = {}
        for mode in ("legacy", "lsq"):
            o = apply_pp(raw, enc, mode)
            o["_mode"] = mode
            report[video][mode] = metrics(o, enc)
            pred = {
                "smpl_params_global": {k: v[0] for k, v in o["pred_smpl_params_global"].items()},
                "smpl_params_incam": {k: v[0] for k, v in o["pred_smpl_params_incam"].items()},
                "K_fullimg": data["K_fullimg"],
            }
            out_dir = Path(cfg.output_dir) / mode
            out_dir.mkdir(exist_ok=True)
            torch.save(detach_to_cpu(pred), out_dir / "hmr4d_results.pt")
            if args.render:
                cfg_m = copy.deepcopy(cfg)
                cfg_m.paths.hmr4d_results = str(out_dir / "hmr4d_results.pt")
                cfg_m.paths.global_video = str(out_dir / "2_global.mp4")
                demo_mod.render_global(cfg_m)
        print(f"\n{video}")
        keys = list(report[video]["legacy"].keys())
        for k in keys:
            a, b = report[video]["legacy"][k], report[video]["lsq"][k]
            fa = "-" if a is None else f"{a:.2f}"
            fb = "-" if b is None else f"{b:.2f}"
            print(f"  {k:18s} legacy {fa:>9s}   lsq {fb:>9s}")
        if args.render:
            demo_mod.merge_videos_horizontal(
                [str(Path(cfg.output_dir) / "legacy" / "2_global.mp4"), str(Path(cfg.output_dir) / "lsq" / "2_global.mp4")],
                str(Path(cfg.output_dir) / "legacy_vs_lsq.mp4"),
            )
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "report.json").write_text(json.dumps(report, indent=1))
    print("\nwrote", Path(args.out) / "report.json")


if __name__ == "__main__":
    main()
