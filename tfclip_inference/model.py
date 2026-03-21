from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from .backbone import DropPath, LayerNorm, QuickGELU, build_visual_backbone, trunc_normal_


def weights_init_kaiming(module: nn.Module) -> None:
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_out")
        nn.init.constant_(module.bias, 0.0)
    elif classname.find("Conv") != -1:
        nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif classname.find("BatchNorm") != -1 and module.affine:
        nn.init.constant_(module.weight, 1.0)
        nn.init.constant_(module.bias, 0.0)


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None, droppath: float = 0.0, T: int = 0):
        super().__init__()
        self.T = T
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
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
        l, bt, d = x.size()
        b = bt // self.T
        x = x.view(l, b, self.T, d)

        msg_token = self.message_fc(x.mean(0))
        msg_token = msg_token.view(b, self.T, 1, d)
        msg_token = msg_token.permute(1, 2, 0, 3).view(self.T, b, d)
        msg_token = msg_token + self.drop_path(
            self.message_attn(
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                need_weights=False,
            )[0]
        )
        msg_token = msg_token.view(self.T, 1, b, d).permute(1, 2, 0, 3)

        x = torch.cat([x, msg_token], dim=0)
        x = x.view(l + 1, -1, d)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TemporalMemoryDiffusion(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, droppath: list[float] | None = None, T: int = 8):
        super().__init__()
        if droppath is None:
            droppath = [0.0 for _ in range(layers)]
        self.resblocks = nn.Sequential(
            *[CrossFramelAttentionBlock(width, heads, droppath=droppath[i], T=T) for i in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.resblocks(x)


class TFClipInferenceModel(nn.Module):
    def __init__(
        self,
        state_dict: dict[str, torch.Tensor],
        backbone: str = "ViT-B-16",
        seq_len: int = 8,
        image_size: tuple[int, int] = (256, 128),
        stride_size: tuple[int, int] = (16, 16),
        neck_feat: str = "before",
        camera_num: int = 0,
        view_num: int = 0,
        sie_camera: bool = False,
        sie_view: bool = False,
        sie_coe: float = 1.0,
    ):
        super().__init__()
        self.model_name = backbone
        self.neck_feat = neck_feat
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_camera = sie_camera
        self.sie_view = sie_view
        self.sie_coe = sie_coe
        self.seq_len = seq_len

        if self.model_name == "ViT-B-16":
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == "RN50":
            self.in_planes = 2048
            self.in_planes_proj = 1024
        else:
            raise ValueError(f"Unsupported backbone: {self.model_name}")

        self.image_encoder = build_visual_backbone(state_dict, image_size=image_size, stride_size=stride_size, backbone_name=backbone)

        if self.sie_camera and self.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif self.sie_camera:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        elif self.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
        else:
            self.cv_embed = None

        self.TMD = TemporalMemoryDiffusion(width=768, layers=1, heads=12, T=seq_len)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

    def _camera_view_embedding(
        self,
        batch_size: int,
        timesteps: int,
        cam_label: torch.Tensor | None,
        view_label: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if self.cv_embed is None:
            return None

        if cam_label is not None and view_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label * self.view_num + view_label]
        elif cam_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[cam_label]
        elif view_label is not None:
            cv_embed = self.sie_coe * self.cv_embed[view_label]
        else:
            return None

        cv_embed = cv_embed.to(device)
        return cv_embed.repeat((1, timesteps)).view(batch_size * timesteps, -1)

    def forward(self, x: torch.Tensor, cam_label: torch.Tensor | None = None, view_label: torch.Tensor | None = None) -> torch.Tensor:
        b, t, c, h, w = x.shape

        if self.model_name == "RN50":
            x = x.view(-1, c, h, w)
            _, image_features, image_features_proj = self.image_encoder(x)
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1)
            img_feature_proj = image_features_proj[0]
            img_feature = img_feature.view(b, t, -1).mean(1)
            img_feature_proj = img_feature_proj.view(b, t, -1).mean(1)
            cls_f_tp = img_feature
        else:
            x = x.view(-1, c, h, w)
            cv_embed = self._camera_view_embedding(b, t, cam_label, view_label, x.device)
            image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)
            img_feature = image_features[:, 0].view(b, t, -1).mean(1)
            img_feature_proj = image_features_proj_raw[:, 0].view(b, t, -1).mean(1)

            ft_for_another_branch = image_features.detach()
            image_features_sat = ft_for_another_branch.permute(1, 0, 2)
            f_sp = self.TMD(image_features_sat)
            f_sp2 = f_sp.permute(1, 0, 2)
            cls_f_sp = f_sp2.mean(1)
            cls_f_tp = cls_f_sp.view(b, t, -1).mean(1)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)

        if self.neck_feat == "after":
            return torch.cat([feat, feat_proj], dim=1)
        return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)
