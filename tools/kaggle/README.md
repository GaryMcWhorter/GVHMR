# Running the webcam comparison on Kaggle

Compares GVHMR's original static-camera post-processing (`legacy`) with this branch's
(`lsq`: joint least-squares root solve plus foot locking) on real footage. The network runs once
per video; both post-processes are applied to the same raw outputs.

## 1. Upload the weights as a private Kaggle dataset

These are licensed, so they have to come from your own downloads (see `docs/INSTALL.md`).
Make a dataset named `gvhmr-inputs` with this layout:

```
checkpoints/
├── body_models/smplx/SMPLX_NEUTRAL.npz
├── body_models/smpl/SMPL_NEUTRAL.pkl
├── gvhmr/gvhmr_siga24_release.ckpt
├── hmr2/epoch=10-step=25000.ckpt
├── vitpose/vitpose-h-multi-coco.pth
└── yolo/yolov8x.pt
```

Webcam clips to test can go in the same dataset (for example `clips/desk_walk.mp4`), or in a
second one.

## 2. Notebook

Settings: GPU (T4 or P100), Internet on, add the `gvhmr-inputs` dataset.

```python
!git clone -b rotor-gv-frame https://github.com/GaryMcWhorter/GVHMR /kaggle/working/GVHMR
%cd /kaggle/working/GVHMR
!bash tools/kaggle/setup_kaggle.sh
```

Quick self-checks (no weights needed, a few seconds):

```python
!python tools/analysis/test_foot_lock.py && python tools/bench/test_rotation_averaging.py
```

Comparison (the tennis clip ships with the repo; add your own with more `--video` flags):

```python
!python tools/kaggle/webcam_compare.py --video docs/example_video/tennis.mp4 \
    --video /kaggle/input/gvhmr-inputs/clips/desk_walk.mp4 --render
```

## 3. What to send back

- `outputs/webcam_compare/report.json` (the numbers)
- `outputs/webcam_compare/<clip>/legacy_vs_lsq.mp4` (side-by-side global view, with `--render`)
- the setup log if anything failed

Good test clips: a full body in view the whole time, walking around and standing still, a
camera that does not move. Turning in place and side steps are where foot sliding shows most.
