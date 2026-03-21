from __future__ import annotations

from os import PathLike
from typing import Sequence, Union

import torch
from PIL import Image

FrameInput = Union[str, PathLike[str], Image.Image, torch.Tensor]
TrackletInput = Sequence[FrameInput]
