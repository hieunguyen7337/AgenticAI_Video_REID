import torch
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_

from .clip_visual import LayerNorm, QuickGELU, load_vit_visual


class CrossFrameAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, droppath=0.0, sequence_length=0):
        super().__init__()
        self.sequence_length = sequence_length

        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        if self.attn_mask is not None:
            self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x):
        tokens, batch_times_time, dim = x.size()
        batch_size = batch_times_time // self.sequence_length
        x = x.view(tokens, batch_size, self.sequence_length, dim)

        msg_token = self.message_fc(x.mean(0))
        msg_token = msg_token.view(batch_size, self.sequence_length, 1, dim)
        msg_token = msg_token.permute(1, 2, 0, 3).view(self.sequence_length, batch_size, dim)
        msg_token = msg_token + self.drop_path(
            self.message_attn(
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                self.message_ln(msg_token),
                need_weights=False,
            )[0]
        )
        msg_token = msg_token.view(self.sequence_length, 1, batch_size, dim).permute(1, 2, 0, 3)

        x = torch.cat([x, msg_token], dim=0)
        x = x.view(tokens + 1, -1, dim)
        x = x + self.drop_path(self.attention(self.ln_1(x)))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        return x


class TransformerSP(nn.Module):
    def __init__(self, width, layers, heads, sequence_length):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[
                CrossFrameAttentionBlock(
                    d_model=width,
                    n_head=heads,
                    droppath=0.0,
                    sequence_length=sequence_length,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x):
        return self.resblocks(x)


class VideoReIDInferenceModel(nn.Module):
    def __init__(self, settings, camera_num=6, view_num=1):
        super().__init__()
        if settings.model.name != "ViT-B-16":
            raise RuntimeError("This inference-only package supports only ViT-B-16.")

        self.settings = settings
        self.in_planes = 768
        self.in_planes_proj = 512
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = settings.model.sie_coe

        height, width = settings.input.size
        stride_height, stride_width = settings.model.stride_size
        h_resolution = int((height - 16) // stride_height + 1)
        w_resolution = int((width - 16) // stride_width + 1)
        vision_stride_size = stride_height

        clip_visual = load_vit_visual(
            pretrained_path=settings.model.pretrain_path,
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            vision_stride_size=vision_stride_size,
        )
        clip_visual.to(settings.model.device)
        self.image_encoder = clip_visual

        if settings.model.sie_camera and settings.model.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print(f"camera number is : {camera_num}")
        elif settings.model.sie_camera:
            self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print(f"camera number is : {camera_num}")
        elif settings.model.sie_view:
            self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
            trunc_normal_(self.cv_embed, std=0.02)
            print(f"camera number is : {view_num}")
        else:
            self.cv_embed = None

        self.sat = TransformerSP(
            width=768,
            layers=1,
            heads=12,
            sequence_length=settings.input.seq_len,
        )

    def _build_cv_embed(self, cam_label, view_label, batch_size, sequence_length):
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

        if cv_embed.dim() == 1:
            cv_embed = cv_embed.unsqueeze(0).expand(batch_size, -1)
        elif cv_embed.size(0) == 1 and batch_size > 1:
            cv_embed = cv_embed.expand(batch_size, -1)
        elif cv_embed.size(0) != batch_size:
            raise RuntimeError(f"Expected cv_embed batch dimension {batch_size}, but got {cv_embed.size(0)}")

        return cv_embed.repeat_interleave(sequence_length, dim=0)

    def forward(self, x, cam_label=None, view_label=None):
        batch_size, sequence_length, channels, height, width = x.shape
        x = x.view(-1, channels, height, width)

        cv_embed = self._build_cv_embed(cam_label, view_label, batch_size, sequence_length)
        image_features, image_features_proj_raw = self.image_encoder(x, cv_embed)

        img_feature = image_features[:, 0].view(batch_size, sequence_length, -1).mean(1)
        img_feature_proj = image_features_proj_raw[:, 0].view(batch_size, sequence_length, -1).mean(1)

        sat_features = self.sat(image_features.detach().permute(1, 0, 2))
        sat_features = sat_features.permute(1, 0, 2).mean(1)
        temporal_feature = sat_features.view(batch_size, sequence_length, -1).mean(1)

        return torch.cat([img_feature, img_feature_proj, temporal_feature], dim=1)

    def load_param(self, trained_path, map_location=None):
        param_dict = torch.load(trained_path, map_location=map_location)
        if isinstance(param_dict, dict) and "state_dict" in param_dict:
            param_dict = param_dict["state_dict"]

        model_state = self.state_dict()
        skipped = []
        for key, value in param_dict.items():
            normalized_key = key.replace("module.", "")
            if normalized_key not in model_state:
                skipped.append(f"{normalized_key} (missing in current model)")
                continue
            if model_state[normalized_key].shape != value.shape:
                skipped.append(
                    f"{normalized_key} (checkpoint {tuple(value.shape)} != model {tuple(model_state[normalized_key].shape)})"
                )
                continue
            model_state[normalized_key].copy_(value)

        print(f"Loading pretrained model from {trained_path}")
        if skipped:
            print("Skipped incompatible parameters:")
            for key in skipped:
                print(f"  - {key}")
