# AgenticAI Video Re-ID Simple Inference

This repository contains a reusable inference package for the CLIP-based video re-identification model, plus a small smoke-test CLI wrapper.

## Inference Package

The reusable inference API lives in [inference/core.py](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/inference/core.py).

Public functions:

- `load_checkpoint_state(model_weight_path, map_location)`
- `infer_num_classes(checkpoint_state)`
- `build_inference_model(model_weight_path, config_path, clip_pretrain_path=None, device=None, camera_num=6, view_num=1)`
- `run_inference(model, video_tensor, cam_label=None, view_label=None, device=None)`

## Inference Input And Output

### `build_inference_model(...)`

Inputs:

- `model_weight_path`: path to the trained re-ID checkpoint, for example `logs_mars/best_model.pth.tar`
- `config_path`: path to the YAML config, by default `configs/vit_clipreid.yml`
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
- `H`: image height, expected to already match the configured test size
- `W`: image width, expected to already match the configured test size

Input requirements:

- `video_tensor` must already be preprocessed and resized before calling `run_inference(...)`
- `video_tensor` must be rank 5
- `cam_label` and `view_label` must match the batch size when provided

Output:

- returns the raw feature tensor produced by the model in eval mode
- for the current config with `TEST.NECK_FEAT: 'before'`, the output is the concatenated feature representation

For the current working smoke test on the A100, the output shape is:

- `torch.Size([2, 2048])`

This means:

- `2` output feature vectors, one per batch item
- `2048` feature dimensions per item

## Smoke-Test CLI Wrapper

The top-level script [simple_inference.py](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/simple_inference.py) is now a thin wrapper around the reusable inference package.

It is intended as a smoke test only:

- it parses CLI arguments
- it builds the inference model
- it creates a local dummy tensor for validation
- it calls `run_inference(...)`
- it prints the device and output feature shape

The reusable package API does not create dummy input internally.

## What The Inference Run Needs

The inference flow expects both of these files:

- `logs_mars/best_model.pth.tar`
- `pretrained/ViT-B-16.pt` or another valid CLIP `ViT-B-16` checkpoint path

`best_model.pth.tar` is the trained re-ID checkpoint. `ViT-B-16.pt` is the CLIP backbone pretrained weight.

## Recommended Environment For A100

Create a fresh environment instead of reusing the older `tfclip` environment.

```bash
conda create -n tfclip_a100 python=3.10 -y
conda activate tfclip_a100
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

If you prefer to install PyTorch first and the rest separately, use:

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install --no-deps -r requirements.txt
```

The `numpy<2` pin is intentional. PyTorch 2.1.2 in this setup should not be used with NumPy 2.x.

## Pretrained CLIP Weight

The model config uses `ViT-B-16`, so inference needs the CLIP `ViT-B-16.pt` checkpoint.

Official checkpoint URL:

- [ViT-B-16.pt](https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt)

Recommended local location:

```bash
mkdir -p pretrained
wget -O pretrained/ViT-B-16.pt https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

If `pretrained/ViT-B-16.pt` exists, the queue script will use it automatically. If it does not exist and `CLIP_PRETRAIN_PATH` is not set, the Python loader will try to auto-download the checkpoint.

## Run On The HPC Queue

Submit the provided queue script:

```bash
qsub scripts/run_simple_inference.sh
```

The script is configured to:

- request 1 GPU
- default to the `tfclip_a100` conda environment
- force inference to run on CUDA
- prefer `pretrained/ViT-B-16.pt` when present
- fall back to CLIP auto-download when no local pretrained weight path is set

You can override the conda environment name if needed:

```bash
CONDA_ENV_NAME=my_env qsub scripts/run_simple_inference.sh
```

## Run Without Queue

For a quick smoke-test run on a login node or any CPU-only session:

```bash
conda activate tfclip_a100
python simple_inference.py --weights logs_mars/best_model.pth.tar --clip-pretrain pretrained/ViT-B-16.pt --device cpu
```

If you want to let the Python loader auto-download the CLIP checkpoint:

```bash
conda activate tfclip_a100
python simple_inference.py --weights logs_mars/best_model.pth.tar --device cpu
```

Use `--device cuda` only when you are inside a GPU-backed session. On many HPC systems, a normal login shell does not have CUDA access, so this command will fail even if the conda environment includes CUDA-enabled PyTorch.

Correct CUDA usage:

- `qsub scripts/run_simple_inference.sh`
- or run `python simple_inference.py ... --device cuda` from an interactive GPU allocation provided by your HPC

If you run from a login node, the correct command is the CPU version above.

## Example Package Usage

```python
import torch

from inference import build_inference_model, run_inference

model, device = build_inference_model(
    model_weight_path="logs_mars/best_model.pth.tar",
    config_path="configs/vit_clipreid.yml",
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

If you want to exercise the package API on GPU, run the same code inside a queued or interactive GPU session and change `device="cpu"` to `device="cuda"`.
