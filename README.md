# AgenticAI Video Re-ID Simple Inference

This repository contains a minimal inference entrypoint for the CLIP-based video re-identification model in this project.

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

## Troubleshooting

### `ModuleNotFoundError: No module named 'ftfy'`

Install the missing tokenizer dependency:

```bash
pip install ftfy regex
```

### PyTorch warns about NumPy 2.x

Downgrade NumPy:

```bash
pip install "numpy<2"
```

### Job lands on CPU instead of GPU

Check that:

- the `#PBS` lines are at the top of `scripts/run_simple_inference.sh`
- the job output shows `CUDA available: True`
- the output includes `Using device: cuda`

If your HPC requires an explicit GPU queue name, submit with that queue as required by your site.

### `CLIP_PRETRAIN_PATH is not set`

That message should no longer appear when `pretrained/ViT-B-16.pt` exists. Put the file there or set:

```bash
export CLIP_PRETRAIN_PATH=/full/path/to/ViT-B-16.pt
```
