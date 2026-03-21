from __future__ import annotations

import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn


def trunc_normal_(x: torch.Tensor, mean: float = 0.0, std: float = 1.0) -> torch.Tensor:
    return x.normal_().fmod_(2).mul_(std).add_(mean)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_type = x.dtype
        out = super().forward(x.float())
        return out.to(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None

        if stride > 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion)),
            ]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int | None = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)
        x, _ = F.multi_head_attention_forward(
            query=x[:1],
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0.0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False,
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    def __init__(self, layers: tuple[int, int, int, int], output_dim: int, heads: int, input_resolution: int, width: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        self._inplanes = width
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32
        self.attnpool = AttentionPool2d(input_resolution, embed_dim, heads, output_dim)

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [Bottleneck(self._inplanes, planes, stride)]
        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        def stem(inp: torch.Tensor) -> torch.Tensor:
            for conv, bn in ((self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)):
                inp = self.relu(bn(conv(inp)))
            return self.avgpool(inp)

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x3)
        xproj = self.attnpool(x4)
        return x3, x4, xproj


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model)),
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=mask)[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int):
        super().__init__()
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        h_resolution: int,
        w_resolution: int,
        patch_size: int,
        stride_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
    ):
        super().__init__()
        scale = width ** -0.5
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=stride_size, bias=False)
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, cv_emb: torch.Tensor | None = None):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1)
        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x11 = self.transformer.resblocks[:11](x)
        x12 = self.transformer.resblocks[11](x11).permute(1, 0, 2)
        x12 = self.ln_post(x12)
        xproj = x12 @ self.proj
        return x12, xproj


def resize_pos_embed(posemb: torch.Tensor, posemb_new: torch.Tensor, height: int, width: int) -> torch.Tensor:
    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(height, width), mode="bilinear")
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, height * width, -1)
    return torch.cat([posemb_token, posemb_grid.squeeze(0)], dim=0)


def _convert_weights_to_fp32(model: nn.Module) -> None:
    def convert(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.float()
            if module.bias is not None:
                module.bias.data = module.bias.data.float()
        if isinstance(module, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(module, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.float()
        if hasattr(module, "proj") and getattr(module, "proj") is not None:
            module.proj.data = module.proj.data.float()

    model.apply(convert)


def _extract_visual_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("image_encoder.") for key in state_dict):
        return {key[len("image_encoder.") :]: value for key, value in state_dict.items() if key.startswith("image_encoder.")}
    if any(key.startswith("visual.") for key in state_dict):
        return {key[len("visual.") :]: value for key, value in state_dict.items() if key.startswith("visual.")}
    return dict(state_dict)


def infer_visual_architecture(state_dict: dict[str, torch.Tensor]) -> str:
    visual_state = _extract_visual_state_dict(state_dict)
    return "ViT-B-16" if "proj" in visual_state else "RN50"


def build_visual_backbone(
    state_dict: dict[str, torch.Tensor],
    image_size: tuple[int, int],
    stride_size: tuple[int, int],
    backbone_name: str | None = None,
) -> nn.Module:
    visual_state = _extract_visual_state_dict(state_dict)
    height, width = image_size
    stride = stride_size[0]
    h_resolution = int((height - 16) // stride + 1)
    w_resolution = int((width - 16) // stride + 1)

    detected_backbone = "ViT-B-16" if "proj" in visual_state else "RN50"
    if backbone_name is not None and backbone_name != detected_backbone:
        raise ValueError(f"Checkpoint backbone is {detected_backbone}, but backbone={backbone_name!r} was requested")

    if detected_backbone == "ViT-B-16":
        vision_width = visual_state["conv1.weight"].shape[0]
        vision_layers = len([k for k in visual_state if k.endswith(".attn.in_proj_weight")])
        patch_size = visual_state["conv1.weight"].shape[-1]
        output_dim = visual_state["proj"].shape[-1]
        heads = vision_width // 64
        visual = VisionTransformer(
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            patch_size=patch_size,
            stride_size=stride,
            width=vision_width,
            layers=vision_layers,
            heads=heads,
            output_dim=output_dim,
        )
        pos_key = "positional_embedding"
        if visual_state[pos_key].shape != visual.positional_embedding.shape:
            visual_state = dict(visual_state)
            visual_state[pos_key] = resize_pos_embed(visual_state[pos_key], visual.positional_embedding, h_resolution, w_resolution)
    else:
        counts = [len(set(k.split(".")[1] for k in visual_state if k.startswith(f"layer{b}."))) for b in (1, 2, 3, 4)]
        vision_layers = tuple(counts)
        vision_width = visual_state["layer1.0.conv1.weight"].shape[0]
        output_width = round((visual_state["attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        output_dim = visual_state["attnpool.c_proj.weight"].shape[0]
        heads = vision_width * 32 // 64
        visual = ModifiedResNet(
            layers=vision_layers,
            output_dim=output_dim,
            heads=heads,
            input_resolution=output_width,
            width=vision_width,
        )
        pos_key = "attnpool.positional_embedding"
        if visual_state[pos_key].shape != visual.attnpool.positional_embedding.shape:
            visual_state = dict(visual_state)
            visual_state[pos_key] = resize_pos_embed(visual_state[pos_key], visual.attnpool.positional_embedding, h_resolution, w_resolution)

    _convert_weights_to_fp32(visual)
    visual.load_state_dict(visual_state, strict=True)
    return visual.eval()
