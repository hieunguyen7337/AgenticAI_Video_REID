import argparse
import os

import torch

from inference import DEFAULT_SETTINGS, build_inference_model, infer_num_classes, load_checkpoint_state, run_inference


def create_smoke_test_input(device):
    batch_size = 2
    seq_len = DEFAULT_SETTINGS.input.seq_len
    height, width = DEFAULT_SETTINGS.input.size

    video_tensor = torch.randn(batch_size, seq_len, 3, height, width, device=device)
    cam_label = torch.tensor([0, 1], device=device)
    view_label = torch.tensor([0, 0], device=device)
    return video_tensor, cam_label, view_label


def run_simple_inference(model_weight_path, clip_pretrain_path=None, device=None):
    checkpoint_state = load_checkpoint_state(model_weight_path, map_location="cpu")
    num_classes = infer_num_classes(checkpoint_state)
    print(f"Building inference model with num_classes={num_classes}")

    model, resolved_device = build_inference_model(
        model_weight_path=model_weight_path,
        clip_pretrain_path=clip_pretrain_path,
        device=device,
    )

    video_tensor, cam_label, view_label = create_smoke_test_input(resolved_device)
    features = run_inference(
        model,
        video_tensor=video_tensor,
        cam_label=cam_label,
        view_label=view_label,
        device=resolved_device,
    )

    if resolved_device == "cuda":
        print(f"Using device: {resolved_device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {resolved_device}")
    print(f"Inference complete. Output feature shape: {features.shape}")
    return features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a simple smoke-test inference pass using the self-contained inference package."
    )
    parser.add_argument(
        "--weights",
        default="logs_mars/best_model.pth.tar",
        help="Path to the trained model weights.",
    )
    parser.add_argument(
        "--clip-pretrain",
        default=os.environ.get("CLIP_PRETRAIN_PATH"),
        help="Path to the CLIP pretrained checkpoint. Overrides CLIP_PRETRAIN_PATH when set.",
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
        clip_pretrain_path=args.clip_pretrain,
        device=args.device,
    )
