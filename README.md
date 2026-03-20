# AgenticAI Video Re-ID Simple Inference

This repository contains a minimal inference entrypoint for the CLIP-based video re-identification model in this project.

## Inference Function

The main entrypoint is `run_simple_inference` in [simple_inference.py](/F:/document/Agentic%20AI_video_re_identification/AgenticAI_Video_REID/simple_inference.py).

Function arguments:

- `model_weight_path`: path to the trained re-ID checkpoint, for example `logs_mars/best_model.pth.tar`
- `config_path`: path to the YAML config, by default `configs/vit_clipreid.yml`
- `clip_pretrain_path`: optional path to the CLIP `ViT-B-16.pt` pretrained checkpoint
- `device`: optional execution device, either `cuda` or `cpu`

Internal model input:

- The function creates a dummy video tensor with shape `(B, T, C, H, W)`
- `B`: batch size
- `T`: sequence length from `cfg.INPUT.SEQ_LEN`
- `C`: 3 RGB channels
- `H`: test image height from `cfg.INPUT.SIZE_TEST[0]`
- `W`: test image width from `cfg.INPUT.SIZE_TEST[1]`

For the default config in this repo, the dummy inference input is:

- `B=2`
- `T=8`
- `C=3`
- `H=256`
- `W=128`

Additional inference inputs:

- `cam_label`: camera IDs for each item in the batch, shape `(B,)`
- `view_label`: view IDs for each item in the batch, shape `(B,)`

Output:

- The function returns a feature tensor for each input tracklet
- In evaluation mode with `TEST.NECK_FEAT: 'before'`, the output is the concatenation of:
- image feature
- projected image feature
- temporal feature

For the current working run on the A100, the output shape is:

- `torch.Size([2, 2048])`

This means:

- `2` output feature vectors, one per batch item
- `2048` feature dimensions per item

## What The Inference Run Needs

The simple inference flow expects both of these files:

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

For a quick manual run:

```bash
conda activate tfclip_a100
python simple_inference.py --weights logs_mars/best_model.pth.tar --clip-pretrain pretrained/ViT-B-16.pt --device cuda
```

If you want to let the Python loader auto-download the CLIP checkpoint:

```bash
conda activate tfclip_a100
python simple_inference.py --weights logs_mars/best_model.pth.tar --device cuda
```
