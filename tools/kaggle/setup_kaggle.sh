#!/usr/bin/env bash
# Set up GVHMR (this fork) on a Kaggle GPU notebook. Run from the repo root.
# Expects the licensed weights as a Kaggle dataset mounted at $GVHMR_INPUTS (default below),
# laid out like inputs/checkpoints in docs/INSTALL.md.
set -euo pipefail
GVHMR_INPUTS=${GVHMR_INPUTS:-/kaggle/input/gvhmr-inputs/checkpoints}

python - <<'PY'
import torch, sys
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "gpu", torch.cuda.is_available())
PY

# Everything in requirements.txt except what Kaggle already ships (torch, torchvision) and what
# needs special handling (pytorch3d, numpy pin, dev tools).
grep -vE '^(--extra-index-url|torch==|torchvision==|pytorch3d|numpy==|jupyter|ipdb|black)' requirements.txt \
  | sed 's/#.*//' | grep -v '^\s*$' > /tmp/req-kaggle.txt
pip install -q "numpy<2" -r /tmp/req-kaggle.txt

# chumpy (needed to read SMPL .pkl files) imports names numpy removed in 1.24; patch them out
CHUMPY_INIT=$(python -c "import importlib.util as u; print(u.find_spec('chumpy').submodule_search_locations[0])")/__init__.py
sed -i 's/^from numpy import bool, int, float, complex, object, unicode, str, nan, inf$/from numpy import nan, inf/' "$CHUMPY_INIT"

# pytorch3d: no wheel for Kaggle's torch, so build it (about 10-15 minutes on a T4 instance)
python -c "import pytorch3d" 2>/dev/null || \
  FORCE_CUDA=1 pip install -q --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"

pip install -q -e . --no-deps

mkdir -p inputs
[ -e inputs/checkpoints ] || ln -s "$GVHMR_INPUTS" inputs/checkpoints
for f in body_models/smplx/SMPLX_NEUTRAL.npz body_models/smpl/SMPL_NEUTRAL.pkl gvhmr/gvhmr_siga24_release.ckpt \
         hmr2/epoch=10-step=25000.ckpt vitpose/vitpose-h-multi-coco.pth yolo/yolov8x.pt; do
  [ -e "inputs/checkpoints/$f" ] && echo "found   $f" || echo "MISSING $f"
done
