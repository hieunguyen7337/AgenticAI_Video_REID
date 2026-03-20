#!/bin/bash -l
set -euo pipefail
#PBS -N tfclip_inference
#PBS -l walltime=02:00:00
#PBS -l mem=16gb
#PBS -l ncpus=1
#PBS -l ngpus=1
#PBS -j eo
#PBS -m abe

echo '================================================'
# Default to current directory if PBS_O_WORKDIR is not set
WORKSPACE="${PBS_O_WORKDIR:-$(pwd)}"
echo "CWD = ${WORKSPACE}"
echo '================================================'
cd "${WORKSPACE}"

# If the script is submitted from within the 'scripts' folder, 
# move up to the standalone_inference root directory
if [[ $(basename "$PWD") == "scripts" ]]; then
    cd ..
fi

echo "Working directory is now: $(pwd)"

echo '=========='
echo 'Load CUDA & cuDNN modules'
echo '=========='
# Retaining module loads from the reference script. 
# Note: tfclip conda env installs cudatoolkit=10.2 via conda.
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

echo '=========='
echo 'Activate conda env'
echo '=========='
source ~/miniconda3/etc/profile.d/conda.sh
# Activate the environment specified in README.md
conda activate tfclip

echo '=========='
echo 'Environment diagnostics'
echo '=========='

nvidia-smi || true
which python

echo "Testing PyTorch import and GPU visibility..."
python - <<EOF
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPUs:", torch.cuda.device_count())
    print("Current GPU:", torch.cuda.get_device_name(0))
    arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
    print("Supported CUDA arch list:", arch_list)
    if arch_list and "sm_80" not in arch_list:
        raise SystemExit("ERROR: This PyTorch build does not support A100 (sm_80). Install a newer CUDA-enabled PyTorch build.")
if not hasattr(torch, "fx"):
    raise SystemExit("ERROR: Installed PyTorch is too old for the current timm package. Upgrade PyTorch or downgrade timm.")
EOF

echo '=========='
echo 'Prepare variables and checks'
echo '=========='

inference_script="simple_inference.py"
model_weight="logs_mars/best_model.pth.tar"
clip_pretrain_path="${CLIP_PRETRAIN_PATH:-}"

# Ensure the inference script exists
if [ ! -s "${inference_script}" ]; then
  echo "ERROR: ${inference_script} was not found! Please make sure you are in the standalone_inference directory."
  exit 1
fi

# Ensure the model weight exists
if [ ! -s "${model_weight}" ]; then
  echo "ERROR: ${model_weight} was not found! Please ensure it's copied into logs_mars/."
  exit 1
fi

if [ -z "${clip_pretrain_path}" ]; then
  echo "ERROR: CLIP_PRETRAIN_PATH is not set."
  echo "Set it to the ViT-B-16.pt checkpoint before running inference."
  exit 1
fi

if [ ! -s "${clip_pretrain_path}" ]; then
  echo "ERROR: CLIP pretrained checkpoint was not found at ${clip_pretrain_path}."
  exit 1
fi

# Respect GPU argument or default to 0
GPU_INDEX="${1:-0}"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

echo "Using GPU index: $CUDA_VISIBLE_DEVICES"

echo '========================='
echo 'Running simple inference'
echo '========================='
date

python "${inference_script}" --weights "${model_weight}" --clip-pretrain "${clip_pretrain_path}" --device cuda

echo '========================='
echo 'Done.'
echo '========================='
date
