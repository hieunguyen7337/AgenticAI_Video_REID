# AgenticAI Video Re-ID Inference-Only Package

This repository is now an inference-only codebase centered on the self-contained [inference](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/inference) package.

## Supported Runtime API

Import only from [inference](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/inference/__init__.py).

Supported public functions:

- `build_inference_model(model_weight_path, clip_pretrain_path=None, device=None, camera_num=6, view_num=1)`
- `run_inference(model, video_tensor, cam_label=None, view_label=None, device=None)`
- `load_checkpoint_state(model_weight_path, map_location="cpu")`
- `infer_num_classes(checkpoint_state)`

The package contains all code needed for inference:

- embedded inference settings
- CLIP `ViT-B-16` visual checkpoint loading
- video re-ID model definition for the working inference path
- checkpoint loading and execution helpers

No runtime import from legacy `config`, `configs`, or `model` paths is required.

## Model Scope

This package supports only the currently working CLIP `ViT-B-16` video re-identification inference path.

It does not support:

- training
- finetuning
- RN50 or other backbone branches
- text or prompt-training code paths

## Inference Input And Output

### `build_inference_model(...)`

Inputs:

- `model_weight_path`: path to the trained re-ID checkpoint, for example `logs_mars/best_model.pth.tar`
- `clip_pretrain_path`: optional path to the CLIP `ViT-B-16.pt` pretrained checkpoint
- `device`: optional execution device, either `cuda` or `cpu`
- `camera_num`: number of cameras for SIE embedding, default `6`
- `view_num`: number of views for SIE embedding, default `1`

Output:

- returns `(model, resolved_device)`
- `model` is a ready-to-run eval-mode model with checkpoint weights loaded
- `resolved_device` is the device string actually used for inference

### `run_inference(...)`

Inputs:

- `model`: the model returned by `build_inference_model(...)`
- `video_tensor`: a prepared tensor with shape `(B, T, C, H, W)`
- `cam_label`: optional integer tensor with shape `(B,)`
- `view_label`: optional integer tensor with shape `(B,)`
- `device`: optional execution device override

Tensor shape meaning:

- `B`: batch size
- `T`: sequence length
- `C`: number of channels, expected `3` for RGB
- `H`: image height, expected `256`
- `W`: image width, expected `128`

Input requirements:

- `video_tensor` must already be preprocessed and resized before calling `run_inference(...)`
- `video_tensor` must be rank `5`
- `cam_label` and `view_label` must match the batch size when provided

Output:

- returns the raw feature tensor produced by the model in eval mode
- the current inference output is the concatenated feature representation
- with the working smoke test, the output shape is `torch.Size([2, 2048])`

## Required Runtime Files

The inference flow expects these runtime artifacts:

- `logs_mars/best_model.pth.tar`
- `pretrained/ViT-B-16.pt` or another valid CLIP `ViT-B-16` checkpoint path

`best_model.pth.tar` is the trained video re-ID checkpoint. `ViT-B-16.pt` is the CLIP visual backbone checkpoint.

## Recommended Environment For A100

```bash
conda create -n tfclip_a100 python=3.10 -y
conda activate tfclip_a100
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

If you prefer to install PyTorch first and the rest separately:

```bash
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -r requirements.txt
```

The `numpy<2` pin is intentional because this environment path should not use NumPy 2.x with the chosen PyTorch build.

## Pretrained CLIP Weight

Official checkpoint URL:

- [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)

Recommended local location:

```bash
mkdir -p pretrained
wget -O pretrained/ViT-B-16.pt https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

If `pretrained/ViT-B-16.pt` exists, the queue script will use it automatically. If it does not exist and `CLIP_PRETRAIN_PATH` is not set, the package will try to auto-download the checkpoint.

## Smoke-Test Wrapper

[simple_inference.py](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/simple_inference.py) is a thin smoke-test wrapper around the package.

It:

- parses CLI arguments
- builds the inference model through `inference`
- creates a local sample tensor
- calls `run_inference(...)`
- prints the device and output shape

### Run Without Queue

On a login node or CPU-only session:

```bash
conda activate tfclip_a100
python simple_inference.py --weights logs_mars/best_model.pth.tar --clip-pretrain pretrained/ViT-B-16.pt --device cpu
```

Use `--device cuda` only inside a GPU-backed session.

### Run On The HPC Queue

```bash
qsub scripts/run_simple_inference.sh
```

The queue script is configured to:

- request 1 GPU
- default to the `tfclip_a100` conda environment
- force inference to run on CUDA
- prefer `pretrained/ViT-B-16.pt` when present
- fall back to CLIP auto-download when no local pretrained weight path is set

## Example Package Usage

```python
import torch

from inference import build_inference_model, run_inference

model, device = build_inference_model(
    model_weight_path="logs_mars/best_model.pth.tar",
    clip_pretrain_path="pretrained/ViT-B-16.pt",
    device="cpu",
)

video_tensor = torch.randn(2, 8, 3, 256, 128)
cam_label = torch.tensor([0, 1])
view_label = torch.tensor([0, 0])

features = run_inference(
    model,
    video_tensor=video_tensor,
    cam_label=cam_label,
    view_label=view_label,
    device=device,
)

print(features.shape)
```
