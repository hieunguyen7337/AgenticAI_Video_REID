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


def _resolve_device(device):
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this environment.")
    return resolved_device


def build_inference_model(
    model_weight_path,
    config_path,
    clip_pretrain_path=None,
    device=None,
    camera_num=6,
    view_num=1,
):
    cfg.merge_from_file(config_path)

    resolved_device = _resolve_device(device)
    cfg.MODEL.DEVICE = resolved_device

    if clip_pretrain_path:
        cfg.MODEL.PRETRAIN_PATH = clip_pretrain_path

    checkpoint_state = load_checkpoint_state(model_weight_path, map_location=resolved_device)
    num_classes = infer_num_classes(checkpoint_state)

    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    model.load_param(model_weight_path, map_location=resolved_device)
    model.eval()
    model.to(resolved_device)
    return model, resolved_device


def run_inference(model, video_tensor, cam_label=None, view_label=None, device=None):
    if video_tensor.dim() != 5:
        raise RuntimeError(
            f"Expected video_tensor to have shape (B, T, C, H, W), but got rank {video_tensor.dim()}"
        )

    resolved_device = _resolve_device(device)
    batch_size = video_tensor.shape[0]

    if cam_label is not None and cam_label.shape[0] != batch_size:
        raise RuntimeError(
            f"Expected cam_label batch dimension {batch_size}, but got {cam_label.shape[0]}"
        )
    if view_label is not None and view_label.shape[0] != batch_size:
        raise RuntimeError(
            f"Expected view_label batch dimension {batch_size}, but got {view_label.shape[0]}"
        )

    video_tensor = video_tensor.to(resolved_device)
    if cam_label is not None:
        cam_label = cam_label.to(resolved_device)
    if view_label is not None:
        view_label = view_label.to(resolved_device)

    with torch.no_grad():
        return model(
            x=video_tensor,
            get_image=False,
            cam_label=cam_label,
            view_label=view_label,
        )
