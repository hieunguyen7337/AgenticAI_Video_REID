from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from tfclip_inference import euclidean_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect saved TF-CLIP embedding files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "test_data_embeddings.pt",
        help="Path to the saved embedding .pt file.",
    )
    parser.add_argument(
        "--preview",
        type=int,
        default=10,
        help="Number of values to print from the start of each embedding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Embedding file not found: {input_path}")

    payload = torch.load(input_path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Expected a dictionary payload saved by main.py")

    tracklets = payload.get("tracklets")
    embeddings = payload.get("embeddings")

    if tracklets is None or embeddings is None:
        raise KeyError("Expected keys 'tracklets' and 'embeddings' in the saved file")

    if not isinstance(embeddings, torch.Tensor):
        raise TypeError("'embeddings' must be a torch.Tensor")

    print(f"File: {input_path}")
    print(f"Embedding tensor shape: {tuple(embeddings.shape)}")
    print(f"Embedding dtype: {embeddings.dtype}")
    print(f"Number of tracklets: {len(tracklets)}")

    for index, name in enumerate(tracklets):
        vector = embeddings[index]
        print("-" * 60)
        print(f"Tracklet {index}: {name}")
        print(f"L2 norm: {vector.norm(p=2).item():.6f}")
        print(f"Min / Max: {vector.min().item():.6f} / {vector.max().item():.6f}")
        print(f"Preview ({args.preview} values): {vector[:args.preview].tolist()}")

    distances = euclidean_distance(embeddings, embeddings)
    print("-" * 60)
    print("Pairwise Euclidean distance matrix:")
    for row_index, row_name in enumerate(tracklets):
        row_values = " ".join(f"{value:.6f}" for value in distances[row_index])
        print(f"{row_name}: {row_values}")

    print("-" * 60)
    print("Pairwise Euclidean distances:")
    for left_index in range(len(tracklets)):
        for right_index in range(left_index + 1, len(tracklets)):
            print(
                f"{tracklets[left_index]} <-> {tracklets[right_index]}: "
                f"{distances[left_index, right_index]:.6f}"
            )


if __name__ == "__main__":
    main()
