import argparse
import os

import torch

from config import cfg
from model.make_model_clipreid import make_model


def load_checkpoint_state(model_weight_path, map_location):
    checkpoint = torch.load(model_weight_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    return checkpoint


def infer_num_classes(checkpoint_state):
    for key in (
        "module.classifier2.weight",
        "classifier2.weight",
        "module.classifier_proj.weight",
        "classifier_proj.weight",
    ):
        if key in checkpoint_state:
            return checkpoint_state[key].shape[0]
    return 100


def run_simple_inference(model_weight_path, config_path, clip_pretrain_path=None, device=None):
    """
    Run a minimal forward pass with dummy video input to validate model loading.
    """
    cfg.merge_from_file(config_path)

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    cfg.MODEL.DEVICE = resolved_device

    if clip_pretrain_path:
        cfg.MODEL.PRETRAIN_PATH = clip_pretrain_path

    checkpoint_state = load_checkpoint_state(model_weight_path, map_location=resolved_device)
    num_classes = infer_num_classes(checkpoint_state)
    print(f"Building inference model with num_classes={num_classes}")

    model = make_model(cfg, num_class=num_classes, camera_num=6, view_num=1)
    model.load_param(model_weight_path, map_location=resolved_device)
    model.eval()
    model.to(resolved_device)

    batch_size = 2
    seq_len = cfg.INPUT.SEQ_LEN
    height, width = cfg.INPUT.SIZE_TEST

    dummy_input = torch.randn(batch_size, seq_len, 3, height, width, device=resolved_device)
    cam_label = torch.tensor([0, 1], device=resolved_device)
    view_label = torch.tensor([0, 0], device=resolved_device)

    with torch.no_grad():
        features = model(
            x=dummy_input,
            get_image=False,
            cam_label=cam_label,
            view_label=view_label,
        )

    if resolved_device == "cuda":
        print(f"Using device: {resolved_device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {resolved_device}")
    print(f"Inference complete. Output feature shape: {features.shape}")
    return features


def parse_args():
    parser = argparse.ArgumentParser(description="Run a simple dummy inference pass.")
    parser.add_argument(
        "--weights",
        default="logs_mars/best_model.pth.tar",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--config",
        default="configs/vit_clipreid.yml",
        help="Path to the experiment config file.",
    )
    parser.add_argument(
        "--clip-pretrain",
        default=os.environ.get("CLIP_PRETRAIN_PATH"),
        help="Path to the CLIP pretrained checkpoint. Overrides config/env when set.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Force inference to run on CPU or CUDA. Defaults to CUDA when available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simple_inference(
        model_weight_path=args.weights,
        config_path=args.config,
        clip_pretrain_path=args.clip_pretrain,
        device=args.device,
    )
