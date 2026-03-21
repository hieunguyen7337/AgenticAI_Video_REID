from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

try:
    import numpy as np
    import torch
    from PIL import Image
except ModuleNotFoundError as exc:
    missing_module = exc.name or "required dependency"
    raise SystemExit(
        f"Missing dependency: {missing_module}. "
        "Install the runtime requirements first: torch, numpy, and Pillow."
    ) from exc

from tfclip_inference import TFClipInferencer


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_CHECKPOINT = Path("logs_mars") / "best_model.pth.tar"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TF-CLIP tracklet inference. "
            "Input can be either a directory of ordered frame images or an explicit list of frame paths."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Either one frame directory or multiple frame image paths in temporal order.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to a TF-CLIP checkpoint. Default: logs_mars/best_model.pth.tar",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda.")
    parser.add_argument("--backbone", default="ViT-B-16", choices=["ViT-B-16", "RN50"])
    parser.add_argument("--seq-len", type=int, default=8, help="Clip length used during inference.")
    parser.add_argument("--height", type=int, default=256, help="Input frame height after resize.")
    parser.add_argument("--width", type=int, default=128, help="Input frame width after resize.")
    parser.add_argument("--cam-id", type=int, default=0, help="Camera id for SIE-enabled checkpoints.")
    parser.add_argument("--view-id", type=int, default=0, help="View id for SIE-enabled checkpoints.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file. Single-tracklet mode saves one embedding; multi-tracklet mode saves all embeddings.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run inference on a synthetic in-memory tracklet instead of reading image files.",
    )
    return parser.parse_args()


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def collect_tracklet(inputs: Sequence[str]) -> list[str]:
    if not inputs:
        raise ValueError("Provide either a frame directory, frame paths, or use --self-test.")

    if len(inputs) == 1:
        candidate = Path(inputs[0])
        if candidate.is_dir():
            frames = sorted(
                path for path in candidate.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if not frames:
                raise ValueError(f"No image frames found in directory: {candidate}")
            return [str(path) for path in frames]

    frame_paths = [Path(path) for path in inputs]
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise ValueError(f"These frame paths do not exist: {missing}")
    invalid = [str(path) for path in frame_paths if path.suffix.lower() not in IMAGE_EXTENSIONS]
    if invalid:
        raise ValueError(f"These inputs are not supported image files: {invalid}")
    return [str(path) for path in frame_paths]


def collect_frames_from_directory(directory: Path) -> list[str]:
    frames = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not frames:
        raise ValueError(f"No image frames found in directory: {directory}")
    return [str(path) for path in frames]


def collect_tracklet_directories(directory: Path) -> list[Path]:
    tracklet_dirs = sorted(path for path in directory.iterdir() if path.is_dir())
    if not tracklet_dirs:
        raise ValueError(f"No tracklet subdirectories found in directory: {directory}")
    return tracklet_dirs


def save_single_embedding(output_path: Path, embedding: torch.Tensor, source: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": source,
        "embedding": embedding.cpu(),
    }
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps({
            "source": source,
            "shape": list(embedding.shape),
            "embedding": embedding.squeeze(0).tolist(),
        }, indent=2))
        return
    torch.save(payload, output_path)


def save_batch_embeddings(output_path: Path, names: Sequence[str], embeddings: torch.Tensor) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tracklets": list(names),
        "embeddings": embeddings.cpu(),
    }
    if output_path.suffix.lower() == ".json":
        output_path.write_text(json.dumps({
            "tracklets": list(names),
            "shape": list(embeddings.shape),
            "embeddings": embeddings.tolist(),
        }, indent=2))
        return
    torch.save(payload, output_path)


def build_synthetic_tracklet(num_frames: int) -> list[Image.Image]:
    frames: list[Image.Image] = []
    for frame_idx in range(num_frames):
        canvas = np.zeros((256, 128, 3), dtype=np.uint8)
        canvas[..., 0] = (frame_idx * 25) % 255
        canvas[..., 1] = np.linspace(0, 255, 128, dtype=np.uint8)[None, :]
        canvas[..., 2] = np.linspace(255, 0, 256, dtype=np.uint8)[:, None]
        frames.append(Image.fromarray(canvas, mode="RGB"))
    return frames


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Place the pretrained .pth/.pth.tar file there or pass --checkpoint explicitly."
        )

    device = args.device or default_device()
    inferencer = TFClipInferencer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        backbone=args.backbone,
        seq_len=args.seq_len,
        image_size=(args.height, args.width),
    )

    if args.self_test:
        tracklet = build_synthetic_tracklet(max(args.seq_len, 8))
        source_description = f"synthetic in-memory tracklet with {len(tracklet)} frames"
        embedding = inferencer.embed_tracklet(tracklet, cam_id=args.cam_id, view_id=args.view_id)
        flattened = embedding.squeeze(0)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device}")
        print(f"Source: {source_description}")
        print(f"Embedding shape: {tuple(embedding.shape)}")
        print(f"Embedding dtype: {embedding.dtype}")
        print(f"Embedding L2 norm: {flattened.norm(p=2).item():.6f}")
        print("Embedding preview (first 10 values):")
        print(flattened[:10].tolist())
        if args.output is not None:
            output_path = args.output.resolve()
            save_single_embedding(output_path, embedding, source_description)
            print(f"Saved embedding to: {output_path}")
    elif len(args.inputs) == 1 and Path(args.inputs[0]).is_dir():
        input_dir = Path(args.inputs[0])
        image_files = [path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if image_files:
            tracklet = collect_frames_from_directory(input_dir)
            source_description = f"{len(tracklet)} frame(s) from input"
            embedding = inferencer.embed_tracklet(tracklet, cam_id=args.cam_id, view_id=args.view_id)
            flattened = embedding.squeeze(0)
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Device: {device}")
            print(f"Source: {source_description}")
            print(f"Embedding shape: {tuple(embedding.shape)}")
            print(f"Embedding dtype: {embedding.dtype}")
            print(f"Embedding L2 norm: {flattened.norm(p=2).item():.6f}")
            print("Embedding preview (first 10 values):")
            print(flattened[:10].tolist())
            if args.output is not None:
                output_path = args.output.resolve()
                save_single_embedding(output_path, embedding, str(input_dir.resolve()))
                print(f"Saved embedding to: {output_path}")
        else:
            tracklet_dirs = collect_tracklet_directories(input_dir)
            tracklet_names = [path.name for path in tracklet_dirs]
            tracklets = [collect_frames_from_directory(path) for path in tracklet_dirs]
            embeddings = inferencer.embed_tracklets(
                tracklets,
                cam_ids=[args.cam_id] * len(tracklets),
                view_ids=[args.view_id] * len(tracklets),
            )
            output_path = (args.output or Path("outputs") / f"{input_dir.name}_embeddings.pt").resolve()
            save_batch_embeddings(output_path, tracklet_names, embeddings)
            print(f"Checkpoint: {checkpoint_path}")
            print(f"Device: {device}")
            print(f"Source: {input_dir.resolve()}")
            print(f"Embedded tracklets: {tracklet_names}")
            print(f"Batch embedding shape: {tuple(embeddings.shape)}")
            print(f"Saved embeddings to: {output_path}")
    else:
        tracklet = collect_tracklet(args.inputs)
        source_description = f"{len(tracklet)} frame(s) from input"
        embedding = inferencer.embed_tracklet(tracklet, cam_id=args.cam_id, view_id=args.view_id)
        flattened = embedding.squeeze(0)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {device}")
        print(f"Source: {source_description}")
        print(f"Embedding shape: {tuple(embedding.shape)}")
        print(f"Embedding dtype: {embedding.dtype}")
        print(f"Embedding L2 norm: {flattened.norm(p=2).item():.6f}")
        print("Embedding preview (first 10 values):")
        print(flattened[:10].tolist())
        if args.output is not None:
            output_path = args.output.resolve()
            save_single_embedding(output_path, embedding, source_description)
            print(f"Saved embedding to: {output_path}")

    if inferencer.load_result.ignored_keys:
        print("Ignored training-only checkpoint keys:")
        for key in inferencer.load_result.ignored_keys:
            print(f"  - {key}")


if __name__ == "__main__":
    main()
