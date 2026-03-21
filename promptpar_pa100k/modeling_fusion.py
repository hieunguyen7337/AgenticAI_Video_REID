from __future__ import annotations

import torch
import torch.nn as nn

from .config import DEFAULT_CONFIG, PromptPARConfig


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None) -> None:
        super().__init__()
        hidden = hidden_features or in_features
        output = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, output)
        self.drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(0.0)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, token_count, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]
        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch_size, token_count, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * 4), out_features=dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        tokenized_attributes: torch.LongTensor,
        config: PromptPARConfig = DEFAULT_CONFIG,
        dim: int = 768,
    ) -> None:
        super().__init__()
        self.config = config
        self.attr_num = tokenized_attributes.shape[0]
        self.dim = dim
        self.word_embed = nn.Linear(clip_model.visual.output_dim, dim)
        self.vis_embed = nn.Linear(clip_model.visual.output_dim, dim)
        self.blocks = nn.ModuleList([Block(dim=dim, num_heads=12) for _ in range(config.mm_layers)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.weight_layer = nn.ModuleList([nn.Linear(dim, 1) for _ in range(self.attr_num)])
        self.bn = nn.BatchNorm1d(self.attr_num)
        self.register_buffer("text_tokens", tokenized_attributes.clone(), persistent=False)
        fusion_len = self.attr_num + 257 + config.vis_prompt
        if not config.use_mm_former:
            self.linear_layer = nn.Linear(fusion_len, self.attr_num)

    def forward(self, imgs: torch.Tensor, clip_model: nn.Module) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size = imgs.shape[0]
        clip_image_features, all_class, _ = clip_model.visual(imgs.type(clip_model.dtype))
        text_tokens = self.text_tokens.to(device=imgs.device)
        text_features = clip_model.encode_text(text_tokens).to(device=imgs.device).float()

        if self.config.use_div:
            final_similarity, _ = clip_model.forward_aggregate(all_class, text_features)
        else:
            final_similarity = None

        textual_features = self.word_embed(text_features).expand(batch_size, self.attr_num, self.dim)
        x = torch.cat([textual_features, self.vis_embed(clip_image_features.float())], dim=1)

        if self.config.use_mm_former:
            for block in self.blocks:
                x = block(x)
        else:
            x = x.permute(0, 2, 1)
            x = self.linear_layer(x)
            x = x.permute(0, 2, 1)

        x = self.norm(x)
        logits = torch.cat([self.weight_layer[index](x[:, index, :]) for index in range(self.attr_num)], dim=1)
        logits = self.bn(logits)
        return logits, final_similarity
