import hashlib
import os
import urllib.request
import warnings
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


VIT_B_16_URL = (
    "https://openaipublic.azureedge.net/clip/models/"
    "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
)


def download_clip_checkpoint(url=VIT_B_16_URL, root=None):
    root = root or os.path.expanduser("~/.cache/clip")
    os.makedirs(root, exist_ok=True)

    filename = os.path.basename(url)
    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        with open(download_target, "rb") as handle:
            if hashlib.sha256(handle.read()).hexdigest() == expected_sha256:
                return download_target
        warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        while True:
            buffer = source.read(8192)
            if not buffer:
                break
            output.write(buffer)

    with open(download_target, "rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != expected_sha256:
            raise RuntimeError("Model has been downloaded but the SHA256 checksum does not match")

    return download_target


def load_clip_checkpoint(pretrained_path=None):
    model_path = pretrained_path or os.environ.get("CLIP_PRETRAIN_PATH")
    if not model_path:
        model_path = download_clip_checkpoint()
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "CLIP pretrained weights were not found. Set clip_pretrain_path or CLIP_PRETRAIN_PATH "
            "to a valid ViT-B-16 checkpoint path."
        )

    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    return state_dict or model.state_dict()


class LayerNorm(nn.LayerNorm):
    def forward(self, x):
        original_dtype = x.dtype
        result = super().forward(x.float())
        return result.to(original_dtype)


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, attn_mask=None):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, h_resolution, w_resolution, patch_size, stride_size, width, layers, heads, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=width,
            kernel_size=patch_size,
            stride=stride_size,
            bias=False,
        )
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x, cv_emb=None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        cls_token = self.class_embedding.to(x.dtype)
        cls_token = cls_token + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls_token, x], dim=1)
        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)
        x11 = self.transformer.resblocks[:11](x)
        x12 = self.transformer.resblocks[11](x11)
        x12 = x12.permute(1, 0, 2)
        x12 = self.ln_post(x12)
        xproj = x12 @ self.proj
        return x12, xproj


def resize_pos_embed(posemb, posemb_new, height, width):
    print("Resized position embedding: %s to %s", posemb.shape, posemb_new.shape)
    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    grid_size_old = int(len(posemb_grid) ** 0.5)
    print(f"Position embedding resize to height:{height} width: {width}")
    posemb_grid = posemb_grid.reshape(1, grid_size_old, grid_size_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)


def build_vit_visual_from_state_dict(state_dict, h_resolution, w_resolution, vision_stride_size):
    if "visual.proj" not in state_dict:
        raise RuntimeError("Only CLIP ViT-B-16 checkpoints are supported in this inference-only package.")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [key for key in state_dict if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")]
    )
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    output_dim = state_dict["text_projection"].shape[1]
    vision_heads = vision_width // 64

    model = VisionTransformer(
        h_resolution=h_resolution,
        w_resolution=w_resolution,
        patch_size=vision_patch_size,
        stride_size=vision_stride_size,
        width=vision_width,
        layers=vision_layers,
        heads=vision_heads,
        output_dim=output_dim,
    )

    visual_state = {}
    for key, value in state_dict.items():
        if key.startswith("visual."):
            visual_state[key.replace("visual.", "", 1)] = value

    visual_state["positional_embedding"] = resize_pos_embed(
        visual_state["positional_embedding"],
        model.positional_embedding,
        h_resolution,
        w_resolution,
    )

    model.load_state_dict(visual_state, strict=True)
    return model.eval()


def load_vit_visual(pretrained_path, h_resolution, w_resolution, vision_stride_size):
    state_dict = load_clip_checkpoint(pretrained_path=pretrained_path)
    return build_vit_visual_from_state_dict(
        state_dict=state_dict,
        h_resolution=h_resolution,
        w_resolution=w_resolution,
        vision_stride_size=vision_stride_size,
    )
