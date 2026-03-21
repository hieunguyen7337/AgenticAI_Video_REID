from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .config import DEFAULT_CONFIG, PromptPARConfig
from .metadata import PA100K_ATTRIBUTES, build_score_mapping
from .modeling_clip import build_model
from .modeling_fusion import TransformerClassifier
from .tokenizer import tokenize


_MODEL_CACHE: dict[tuple[str, str], tuple[torch.nn.Module, torch.nn.Module]] = {}
_PREPROCESS = transforms.Compose(
    [
        transforms.Resize((DEFAULT_CONFIG.image_size, DEFAULT_CONFIG.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def _strip_module_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        normalized_key = key[7:] if key.startswith("module.") else key
        normalized_key = normalized_key.replace("visual_embed.", "vis_embed.")
        cleaned[normalized_key] = value
    return cleaned


def _load_checkpoint(config: PromptPARConfig) -> dict[str, Any]:
    checkpoint_path = config.checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at '{checkpoint_path}'.")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint payload at '{checkpoint_path}'. Expected a dictionary.")
    return checkpoint


def _extract_clip_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    clip_state = checkpoint.get("ViT_model") or checkpoint.get("clip_model")
    if clip_state is None:
        raise KeyError("Checkpoint must contain either 'ViT_model' or 'clip_model'.")
    return _strip_module_prefix(clip_state)


def _extract_classifier_state(checkpoint: dict[str, Any]) -> dict[str, torch.Tensor]:
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Checkpoint must contain 'model_state_dict' for the PromptPAR classifier head.")
    return _strip_module_prefix(state)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _cache_key(device: torch.device, config: PromptPARConfig) -> tuple[str, str]:
    return str(device), str(config.checkpoint_path.resolve())


def _build_bundle(device: torch.device, config: PromptPARConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
    checkpoint = _load_checkpoint(config)
    clip_model = build_model(_extract_clip_state(checkpoint), config=config)
    tokenized_attributes = tokenize(PA100K_ATTRIBUTES, context_length=config.context_length)
    model = TransformerClassifier(clip_model=clip_model, tokenized_attributes=tokenized_attributes, config=config)
    classifier_state = _extract_classifier_state(checkpoint)
    incompatible = model.load_state_dict(classifier_state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(
            "Classifier checkpoint is missing required keys: " + ", ".join(sorted(incompatible.missing_keys[:10]))
        )

    clip_model = clip_model.to(device)
    model = model.to(device)
    clip_model.eval()
    model.eval()
    if device.type == "cpu":
        clip_model.float()
        model.float()
    return model, clip_model


def _get_bundle(device: torch.device, config: PromptPARConfig) -> tuple[torch.nn.Module, torch.nn.Module]:
    key = _cache_key(device, config)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = _build_bundle(device=device, config=config)
    return _MODEL_CACHE[key]


def _prepare_pil_image(image: str | Path | Image.Image | np.ndarray) -> Image.Image:
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = image
        if array.ndim == 2:
            array = np.stack([array] * 3, axis=-1)
        if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
            raise ValueError("NumPy inputs must be HxW, HxWx1, HxWx3, or HxWx4.")
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        if array.shape[-1] == 4:
            array = array[..., :3]
        if array.dtype != np.uint8:
            array = np.clip(array, 0.0, 255.0)
            if array.max() <= 1.0:
                array = array * 255.0
            array = array.astype(np.uint8)
        return Image.fromarray(array).convert("RGB")
    raise TypeError("Image must be a path, PIL.Image, NumPy array, or torch.Tensor.")


def _prepare_tensor_image(image: torch.Tensor, config: PromptPARConfig) -> torch.Tensor:
    tensor = image.detach().clone()
    if tensor.ndim == 4:
        if tensor.shape[0] != 1:
            raise ValueError("Tensor input must represent a single image.")
        tensor = tensor.squeeze(0)
    if tensor.ndim != 3:
        raise ValueError("Tensor input must be 3D (C,H,W) or (H,W,C).")
    if tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    if tensor.shape[0] != 3:
        raise ValueError("Tensor input must have 3 channels.")

    tensor = tensor.float()
    if tensor.max().item() > 1.0:
        tensor = tensor / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=(config.image_size, config.image_size), mode="bilinear", align_corners=False)
    tensor = tensor.squeeze(0)
    mean = torch.tensor([0.5, 0.5, 0.5], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _prepare_image_tensor(image: Any, device: torch.device, config: PromptPARConfig) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = _prepare_tensor_image(image, config)
    else:
        pil_image = _prepare_pil_image(image)
        tensor = _PREPROCESS(pil_image)
    return tensor.unsqueeze(0).to(device)


def infer_attributes(
    image: Any,
    threshold: float = DEFAULT_CONFIG.default_threshold,
    device: str | torch.device | None = None,
    return_probs: bool = False,
) -> dict[str, Any]:
    config = DEFAULT_CONFIG
    device_obj = _resolve_device(device)
    model, clip_model = _get_bundle(device=device_obj, config=config)
    image_tensor = _prepare_image_tensor(image=image, device=device_obj, config=config)

    with torch.no_grad():
        logits, _ = model(image_tensor, clip_model=clip_model)
        probabilities = torch.sigmoid(logits)[0].detach().cpu().tolist()

    scores: OrderedDict[str, float] = build_score_mapping(probabilities)
    predicted_attributes = [name for name, score in scores.items() if score > threshold]
    result = {
        "attributes": predicted_attributes,
        "scores": scores,
        "threshold": float(threshold),
    }
    if return_probs:
        result["probabilities"] = scores
    return result
