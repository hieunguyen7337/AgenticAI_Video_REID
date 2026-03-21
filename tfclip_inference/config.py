from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TFClipConfig:
    checkpoint_name: str = "best_model.pth.tar"
    backbone: str = "ViT-B-16"
    seq_len: int = 8
    image_height: int = 256
    image_width: int = 128
    stride_height: int = 16
    stride_width: int = 16
    neck_feat: str = "before"
    normalize: bool = True
    camera_num: int = 0
    view_num: int = 0
    sie_camera: bool = False
    sie_view: bool = False
    sie_coe: float = 1.0

    @property
    def package_root(self) -> Path:
        return Path(__file__).resolve().parent

    @property
    def weights_dir(self) -> Path:
        return self.package_root / "weights"

    @property
    def checkpoint_path(self) -> Path:
        return self.weights_dir / self.checkpoint_name

    @property
    def image_size(self) -> tuple[int, int]:
        return self.image_height, self.image_width

    @property
    def stride_size(self) -> tuple[int, int]:
        return self.stride_height, self.stride_width


DEFAULT_CONFIG = TFClipConfig()
