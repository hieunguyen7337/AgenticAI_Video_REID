import torch
import torch.nn as nn
from collections import OrderedDict
from timm.layers import DropPath, trunc_normal_

from .clip_visual import LayerNorm, QuickGELU, load_vit_visual


def weights_init_kaiming(module):
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


def weights_init_classifier(module):
    classname = module.__class__.__name__
    if classname.find("Linear") != -1:
        nn.init.normal_(module.weight, std=0.001)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class CrossFramelAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None, droppath=0.0, T=0):
        super().__init__()
        self.T = T
        self.message_fc = nn.Linear(d_model, d_model)
        self.message_ln = LayerNorm(d_model)
        self.message_attn = nn.MultiheadAttention(d_model, n_head)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.drop_path = DropPath(droppath) if droppath > 0.0 else nn.Identity()
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


class Transformer_SP(nn.Module):
    def __init__(self, width, layers, heads, T):
        super().__init__()
        self.resblocks = nn.Sequential(
            *[CrossFramelAttentionBlock(width, heads, droppath=0.0, T=T) for _ in range(layers)]
        )

    def forward(self, x):
        return self.resblocks(x)


class MulitHeadAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v):
        batch_size, q_tokens, channels = q.shape
        _, k_tokens, _ = k.shape
        q = self.q_proj(q).reshape(batch_size, q_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(batch_size, k_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(batch_size, k_tokens, self.num_heads, channels // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, q_tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PromptGeneratorLayer(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.self_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.cross_attn = MulitHeadAttention(d_model, nhead, proj_drop=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            QuickGELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, visual):
        q = k = v = self.norm1(x)
        x = x + self.self_attn(q, k, v)
        q = self.norm2(x)
        x = x + self.cross_attn(q, visual, visual)
        x = x + self.dropout(self.mlp(self.norm3(x)))
        return x


class ImageSpecificPrompt(nn.Module):
    def __init__(self, layers=2, embed_dim=512, alpha=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.memory_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )
        self.text_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )
        self.decoder = nn.ModuleList([PromptGeneratorLayer(embed_dim, embed_dim // 64) for _ in range(layers)])
        self.alpha = nn.Parameter(torch.ones(embed_dim) * alpha)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, text, visual):
        visual = self.memory_proj(visual)
        text = self.text_proj(text)
        for layer in self.decoder:
            text = layer(text, visual)
        text = self.out_proj(text)
        return text


class VideoReIDInferenceModel(nn.Module):
    def __init__(self, settings, num_classes, camera_num=6, view_num=1):
        super().__init__()
        if settings.model.name != "ViT-B-16":
            raise RuntimeError("This inference-only package supports only ViT-B-16.")

        self.settings = settings
        self.model_name = settings.model.name
        self.neck_feat = settings.test.neck_feat
        self.in_planes = 768
        self.in_planes_proj = 512
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = settings.model.sie_coe

        self.classifier2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier2.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)
        self.classifier_proj_temp = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp.apply(weights_init_classifier)
        self.classifier_proj_temp2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_proj_temp2.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)
        self.bottleneck_proj_temp = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp.bias.requires_grad_(False)
        self.bottleneck_proj_temp.apply(weights_init_kaiming)
        self.bottleneck_proj_temp2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_proj_temp2.bias.requires_grad_(False)
        self.bottleneck_proj_temp2.apply(weights_init_kaiming)

        height, width = settings.input.size
        stride_height, stride_width = settings.model.stride_size
        self.h_resolution = int((height - 16) // stride_height + 1)
        self.w_resolution = int((width - 16) // stride_width + 1)
        self.vision_stride_size = stride_height

        clip_visual = load_vit_visual(
            pretrained_path=settings.model.pretrain_path,
            h_resolution=self.h_resolution,
            w_resolution=self.w_resolution,
            vision_stride_size=self.vision_stride_size,
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

        self.prompts_generator = ImageSpecificPrompt()
        self.SAT = Transformer_SP(width=768, layers=1, heads=12, T=settings.input.seq_len)

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

        image_features_sat = image_features.detach().permute(1, 0, 2)
        sat_features = self.SAT(image_features_sat)
        sat_features = sat_features.permute(1, 0, 2)
        cls_f_sp = sat_features.mean(1)
        cls_f_tp = cls_f_sp.view(batch_size, sequence_length, -1).mean(1)

        feat = self.bottleneck(img_feature)
        feat_proj = self.bottleneck_proj(img_feature_proj)
        _ = self.bottleneck_proj_temp(cls_f_sp)
        _ = self.bottleneck_proj_temp2(cls_f_tp)

        if self.neck_feat == "after":
            return torch.cat([feat, feat_proj], dim=1)
        return torch.cat([img_feature, img_feature_proj, cls_f_tp], dim=1)

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
