from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .config import DEFAULT_CONFIG, TFClipConfig
from .model import TFClipInferenceModel
from .preprocess import DEFAULT_IMAGE_SIZE, preprocess_clip
from .sampling import dense_sample_frames
from .types import TrackletInput


@dataclass(frozen=True)
class LoadResult:
    ignored_keys: tuple[str, ...]


TRAINING_ONLY_PREFIXES = (
    "classifier",
    "classifier2",
    "classifier_proj",
    "classifier_proj_temp",
    "classifier_proj_temp2",
    "bottleneck_proj_temp",
    "bottleneck_proj_temp2",
    "SSP",
)


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a state_dict or a dict containing one")
    return {key.replace("module.", "", 1): value for key, value in checkpoint.items()}


def _remap_state_dict_for_inference(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            remapped[f"image_encoder.{key[len('visual.') :]}"] = value
        else:
            remapped[key] = value
    return remapped


class TFClipInferencer:
    def __init__(
        self,
        model: TFClipInferenceModel,
        device: str | torch.device = "cpu",
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        seq_len: int = 8,
        normalize: bool = True,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.image_size = image_size
        self.seq_len = seq_len
        self.normalize = normalize
        self.load_result = LoadResult(ignored_keys=tuple())

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
        backbone: str = "ViT-B-16",
        seq_len: int = 8,
        image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
        stride_size: tuple[int, int] = (16, 16),
        neck_feat: str = "before",
        normalize: bool = True,
        camera_num: int = 0,
        view_num: int = 0,
        sie_camera: bool = False,
        sie_view: bool = False,
        sie_coe: float = 1.0,
    ) -> "TFClipInferencer":
        checkpoint = torch.load(Path(checkpoint_path), map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        model = TFClipInferenceModel(
            state_dict=state_dict,
            backbone=backbone,
            seq_len=seq_len,
            image_size=image_size,
            stride_size=stride_size,
            neck_feat=neck_feat,
            camera_num=camera_num,
            view_num=view_num,
            sie_camera=sie_camera,
            sie_view=sie_view,
            sie_coe=sie_coe,
        )
        remapped_state_dict = _remap_state_dict_for_inference(state_dict)
        incompatible = model.load_state_dict(remapped_state_dict, strict=False)
        ignored = tuple(sorted(k for k in incompatible.unexpected_keys if k.startswith(TRAINING_ONLY_PREFIXES)))
        inferencer = cls(model=model, device=device, image_size=image_size, seq_len=seq_len, normalize=normalize)
        inferencer.load_result = LoadResult(ignored_keys=ignored)
        return inferencer

    @classmethod
    def from_packaged_checkpoint(
        cls,
        device: str | torch.device = "cpu",
        config: TFClipConfig = DEFAULT_CONFIG,
        **overrides,
    ) -> "TFClipInferencer":
        checkpoint_path = config.checkpoint_path
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Packaged TF-CLIP checkpoint not found at {checkpoint_path}. "
                "Keep the weights/ directory with the tfclip_inference package."
            )

        return cls.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            backbone=overrides.get("backbone", config.backbone),
            seq_len=overrides.get("seq_len", config.seq_len),
            image_size=overrides.get("image_size", config.image_size),
            stride_size=overrides.get("stride_size", config.stride_size),
            neck_feat=overrides.get("neck_feat", config.neck_feat),
            normalize=overrides.get("normalize", config.normalize),
            camera_num=overrides.get("camera_num", config.camera_num),
            view_num=overrides.get("view_num", config.view_num),
            sie_camera=overrides.get("sie_camera", config.sie_camera),
            sie_view=overrides.get("sie_view", config.sie_view),
            sie_coe=overrides.get("sie_coe", config.sie_coe),
        )

    def _maybe_normalize(self, embedding: torch.Tensor, normalize: bool) -> torch.Tensor:
        if not normalize:
            return embedding
        return torch.nn.functional.normalize(embedding, dim=1, p=2)

    def embed_clip(self, clip_tensor: torch.Tensor, cam_id: int = 0, view_id: int = 0, normalize: bool | None = None) -> torch.Tensor:
        if clip_tensor.ndim != 4:
            raise ValueError("clip_tensor must have shape (T, C, H, W)")

        batched = clip_tensor.unsqueeze(0).to(self.device)
        cam_label = torch.tensor([cam_id], dtype=torch.long, device=self.device) if self.model.sie_camera else None
        view_label = torch.tensor([view_id], dtype=torch.long, device=self.device) if self.model.sie_view else None
        apply_normalize = self.normalize if normalize is None else normalize

        with torch.no_grad():
            embedding = self.model(batched, cam_label=cam_label, view_label=view_label).cpu()
        return self._maybe_normalize(embedding, apply_normalize)

    def embed_tracklet(self, frames: TrackletInput, cam_id: int = 0, view_id: int = 0) -> torch.Tensor:
        clips = dense_sample_frames(frames, self.seq_len)
        clip_embeddings = []
        for clip_frames in clips:
            clip_tensor = preprocess_clip(clip_frames, image_size=self.image_size)
            clip_embeddings.append(self.embed_clip(clip_tensor, cam_id=cam_id, view_id=view_id, normalize=False))
        embedding = torch.stack(clip_embeddings, dim=0).mean(dim=0)
        return self._maybe_normalize(embedding, self.normalize)

    def embed_tracklets(
        self,
        tracklets: Sequence[TrackletInput],
        cam_ids: Sequence[int] | None = None,
        view_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        if cam_ids is None:
            cam_ids = [0] * len(tracklets)
        if view_ids is None:
            view_ids = [0] * len(tracklets)
        if len(tracklets) != len(cam_ids) or len(tracklets) != len(view_ids):
            raise ValueError("tracklets, cam_ids, and view_ids must have the same length")

        embeddings = [
            self.embed_tracklet(tracklet, cam_id=cam_id, view_id=view_id)
            for tracklet, cam_id, view_id in zip(tracklets, cam_ids, view_ids)
        ]
        return torch.cat(embeddings, dim=0)
