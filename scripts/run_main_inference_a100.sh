#!/bin/bash -l
#PBS -N tfclip_main_infer
#PBS -l walltime=02:00:00
#PBS -l mem=16gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

set -euo pipefail

echo '================================================'
WORKSPACE="${PBS_O_WORKDIR:-$(pwd)}"
echo "Submission directory = ${WORKSPACE}"
echo '================================================'
cd "${WORKSPACE}"

if [[ $(basename "$PWD") == "script" ]]; then
    cd ..
fi

echo "Working directory is now: $(pwd)"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tfclip_a100}"
echo "Activating conda env: ${CONDA_ENV_NAME}"
conda activate "${CONDA_ENV_NAME}"

echo '=========='
echo 'Environment diagnostics'
echo '=========='
nvidia-smi || true
which python

python - <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    print("Supported CUDA arch list:", arch_list)
    if arch_list and "sm_80" not in arch_list:
        raise SystemExit("ERROR: This PyTorch build does not support NVIDIA A100 (sm_80).")
EOF

echo '=========='
echo 'Prepare inference arguments'
echo '=========='

PYTHON_SCRIPT="main.py"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-logs_mars/best_model.pth.tar}"
TRACKLET_INPUT="${TRACKLET_INPUT:-}"
BACKBONE="${BACKBONE:-ViT-B-16}"
SEQ_LEN="${SEQ_LEN:-8}"
HEIGHT="${HEIGHT:-256}"
WIDTH="${WIDTH:-128}"
CAM_ID="${CAM_ID:-0}"
VIEW_ID="${VIEW_ID:-0}"
GPU_INDEX="${1:-0}"

if [ ! -s "${PYTHON_SCRIPT}" ]; then
  echo "ERROR: ${PYTHON_SCRIPT} was not found in $(pwd)."
  exit 1
fi

if [ ! -s "${CHECKPOINT_PATH}" ]; then
  echo "ERROR: checkpoint not found at ${CHECKPOINT_PATH}."
  echo "Place the TF-CLIP checkpoint there or export CHECKPOINT_PATH=/path/to/best_model.pth.tar"
  exit 1
fi

if [ -z "${TRACKLET_INPUT}" ]; then
  echo "ERROR: TRACKLET_INPUT is not set."
  echo "Set TRACKLET_INPUT to either a directory of ordered frames or one image path."
  echo "Example:"
  echo "  qsub -v TRACKLET_INPUT=data/query_tracklet script/run_main_inference_a100.sh"
  exit 1
fi

if [ ! -e "${TRACKLET_INPUT}" ]; then
  echo "ERROR: TRACKLET_INPUT does not exist: ${TRACKLET_INPUT}"
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU_INDEX}"

echo "Using GPU index: ${CUDA_VISIBLE_DEVICES}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Tracklet input: ${TRACKLET_INPUT}"

echo '========================='
echo 'Running TF-CLIP main.py'
echo '========================='
date

python "${PYTHON_SCRIPT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --device cuda \
  --backbone "${BACKBONE}" \
  --seq-len "${SEQ_LEN}" \
  --height "${HEIGHT}" \
  --width "${WIDTH}" \
  --cam-id "${CAM_ID}" \
  --view-id "${VIEW_ID}" \
  "${TRACKLET_INPUT}"

echo '========================='
echo 'Done.'
echo '========================='
date
