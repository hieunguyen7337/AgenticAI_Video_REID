from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image

from .types import FrameInput

DEFAULT_IMAGE_SIZE = (256, 128)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def load_frame(frame: FrameInput) -> Image.Image:
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")

    if isinstance(frame, (str, Path)):
        return Image.open(frame).convert("RGB")

    if isinstance(frame, torch.Tensor):
        return tensor_to_pil(frame)

    raise TypeError(f"Unsupported frame type: {type(frame)!r}")


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    tensor = frame.detach().cpu()
    if tensor.ndim != 3:
        raise ValueError("Tensor frame must have shape (C, H, W) or (H, W, C)")

    if tensor.shape[0] in (1, 3):
        chw = tensor
    elif tensor.shape[-1] in (1, 3):
        chw = tensor.permute(2, 0, 1)
    else:
        raise ValueError("Tensor frame must have 1 or 3 channels")

    chw = chw.float()
    if chw.max() <= 1.0 and chw.min() >= 0.0:
        chw = chw * 255.0
    chw = chw.clamp(0, 255).byte()

    if chw.shape[0] == 1:
        chw = chw.repeat(3, 1, 1)

    hwc = chw.permute(1, 2, 0).numpy()
    return Image.fromarray(hwc, mode="RGB")


def preprocess_frame(
    frame: FrameInput,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Iterable[float] = DEFAULT_MEAN,
    std: Iterable[float] = DEFAULT_STD,
) -> torch.Tensor:
    image = load_frame(frame)
    height, width = image_size
    if image.size != (width, height):
        image = image.resize((width, height), Image.BILINEAR)

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean_tensor = torch.tensor(tuple(mean), dtype=tensor.dtype).view(3, 1, 1)
    std_tensor = torch.tensor(tuple(std), dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean_tensor) / std_tensor


def preprocess_clip(
    frames: Iterable[FrameInput],
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
    mean: Iterable[float] = DEFAULT_MEAN,
    std: Iterable[float] = DEFAULT_STD,
) -> torch.Tensor:
    tensors = [preprocess_frame(frame, image_size=image_size, mean=mean, std=std) for frame in frames]
    if not tensors:
        raise ValueError("frames must not be empty")
    return torch.stack(tensors, dim=0)
