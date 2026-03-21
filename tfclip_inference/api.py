from __future__ import annotations

from typing import Sequence

import torch

from .config import DEFAULT_CONFIG, TFClipConfig
from .inferencer import TFClipInferencer
from .types import TrackletInput


_INFERENCER_CACHE: dict[tuple[str, str, str, int, tuple[int, int]], TFClipInferencer] = {}


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cache_key(device: torch.device, config: TFClipConfig) -> tuple[str, str, str, int, tuple[int, int]]:
    return (
        str(device),
        str(config.checkpoint_path.resolve()),
        config.backbone,
        config.seq_len,
        config.image_size,
    )


def load_inferencer(
    device: str | torch.device | None = None,
    config: TFClipConfig = DEFAULT_CONFIG,
) -> TFClipInferencer:
    device_obj = resolve_device(device)
    key = _cache_key(device_obj, config)
    if key not in _INFERENCER_CACHE:
        _INFERENCER_CACHE[key] = TFClipInferencer.from_packaged_checkpoint(device=device_obj, config=config)
    return _INFERENCER_CACHE[key]


def infer_tracklet(
    frames: TrackletInput,
    cam_id: int = 0,
    view_id: int = 0,
    device: str | torch.device | None = None,
    config: TFClipConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    inferencer = load_inferencer(device=device, config=config)
    return inferencer.embed_tracklet(frames, cam_id=cam_id, view_id=view_id)


def infer_tracklets(
    tracklets: Sequence[TrackletInput],
    cam_ids: Sequence[int] | None = None,
    view_ids: Sequence[int] | None = None,
    device: str | torch.device | None = None,
    config: TFClipConfig = DEFAULT_CONFIG,
) -> torch.Tensor:
    inferencer = load_inferencer(device=device, config=config)
    return inferencer.embed_tracklets(tracklets, cam_ids=cam_ids, view_ids=view_ids)
