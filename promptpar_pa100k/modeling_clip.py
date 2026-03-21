from __future__ import annotations

import math
from collections import OrderedDict
from functools import reduce
from operator import mul

import numpy as np
import torch
from torch import nn

from .config import DEFAULT_CONFIG, PromptPARConfig


class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_dtype = x.dtype
        return super().forward(x.float()).to(original_dtype)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor | None = None) -> None:
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

    def attention(self, x: torch.Tensor, visual_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_mask = self.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)
        if visual_mask is not None:
            attn_mask = visual_mask.to(dtype=x.dtype, device=x.device)
        return self.attn(x, x, x, need_weights=True, attn_mask=attn_mask)

    def forward(self, x: torch.Tensor, visual_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.attention(self.ln_1(x), visual_mask)
        x = x + attn_output
        x = x + self.mlp(self.ln_2(x))
        return x, attn_weights


class Transformer(nn.Module):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        config: PromptPARConfig,
        attr_count: int,
        attn_mask: torch.Tensor | None = None,
        visual_or_text: bool = False,
        prompt_num: int = 25,
        part_num: list[int] | None = None,
        row_patch_num: int = 0,
    ) -> None:
        super().__init__()
        self.config = config
        self.visual_or_text = visual_or_text
        self.width = width
        self.layers = layers
        self.prompt_num = prompt_num
        self.part_num = part_num or []
        self.div_prompt_num = int(prompt_num / (config.div_num + 1)) if config.div_num else 0
        self.vis_len = prompt_num + 5 + row_patch_num**2
        self.prefix_len = prompt_num + 5

        init_scale = math.sqrt(6.0 / float(3 * reduce(mul, (14, 14), 1) + width))
        if self.visual_or_text:
            self.prompt_deep = nn.Parameter(torch.zeros(config.vis_depth, prompt_num, 1, width))
            nn.init.uniform_(self.prompt_deep.data, -init_scale, init_scale)
        else:
            self.prompt_text_deep = nn.Parameter(torch.zeros(layers, prompt_num, attr_count, width))
            nn.init.uniform_(self.prompt_text_deep.data, -init_scale, init_scale)

        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])
        if self.visual_or_text and config.use_div and config.use_vismask:
            self.visual_mask = self.build_visual_mask()
        else:
            self.visual_mask = None

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[1]
        if self.visual_or_text:
            attn_output_weights: torch.Tensor | None = None
            for layer_index, block in enumerate(self.resblocks):
                if self.config.use_div:
                    if layer_index < self.config.vis_depth:
                        for div_index in range(self.config.div_num + 1):
                            start_idx = 1 if div_index == 0 else self.div_prompt_num * div_index + div_index + 1
                            prompts = self.prompt_deep[layer_index, self.div_prompt_num * div_index : self.div_prompt_num * (div_index + 1)]
                            tail_start = start_idx if layer_index == 0 else start_idx + self.div_prompt_num
                            x = torch.cat(
                                [x[:start_idx], prompts.repeat(1, batch_size, 1).to(x.device).to(x.dtype), x[tail_start:]],
                                dim=0,
                            )
                elif layer_index < self.config.vis_depth:
                    x = torch.cat(
                        [
                            x[:1],
                            self.prompt_deep[layer_index].repeat(1, batch_size, 1).to(x.device).to(x.dtype),
                            x[1 if layer_index == 0 else self.prompt_num + 1 :],
                        ],
                        dim=0,
                    )
                x, attn_output_weights = block(x, self.visual_mask)

            all_class = None
            if self.config.use_div:
                all_class = torch.stack(
                    [
                        x[1 if index == 0 else self.div_prompt_num * index + index]
                        for index in range(self.config.div_num + 1)
                    ]
                ).permute(1, 0, 2)
            return x, all_class, attn_output_weights

        for layer_index, block in enumerate(self.resblocks):
            if self.config.use_textprompt:
                if layer_index == 0:
                    x = torch.cat([x, self.prompt_text_deep[0].to(x.device).to(x.dtype)], dim=0)
                x = torch.cat([x[:77, :, :], self.prompt_text_deep[layer_index].to(x.device).to(x.dtype)], dim=0)
            x, _ = block(x)
        return x

    def build_visual_mask(self) -> torch.Tensor:
        visual_mask = []
        length = self.vis_len
        every_prefix_len = 1 + self.div_prompt_num
        mask_part_start = [self.prefix_len, self.prefix_len + 48, self.prefix_len + 112, self.prefix_len + 176]
        masked_rows = [index for index in range(1 + self.div_prompt_num, self.prefix_len)]
        for row_index in range(length):
            attn_mask = torch.zeros(length, dtype=torch.float)
            if row_index in masked_rows:
                part_idx = int(row_index / every_prefix_len)
                mask_class_prefix = [index for index in range(part_idx * every_prefix_len)]
                mask_class_suffix = [index for index in range((part_idx + 1) * every_prefix_len, self.prefix_len)]
                all_patch_indices = [index for index in range(self.prefix_len, length)]
                visible_patch_indices = [
                    index for index in range(mask_part_start[part_idx - 1], mask_part_start[part_idx - 1] + self.part_num[part_idx - 1])
                ]
                masked_patch_indices = [index for index in all_patch_indices if index not in visible_patch_indices]
                attn_mask[mask_class_prefix + mask_class_suffix + masked_patch_indices] = float("-inf")
            visual_mask.append(attn_mask)
        return torch.stack(visual_mask)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        output_dim: int,
        prompt_num: int,
        config: PromptPARConfig,
    ) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        row_patch_num = input_resolution // patch_size
        base_part_row_num = int(row_patch_num // config.div_num) + int(config.overlap_row // 2)
        self.part_row_num = [
            base_part_row_num + int(config.overlap_row // 2) if row not in (0, config.div_num - 1) else base_part_row_num
            for row in range(config.div_num)
        ]
        self.part_num = [part_row * row_patch_num for part_row in self.part_row_num]
        if config.use_div:
            self.part_class_embedding = nn.Parameter(scale * torch.randn((config.div_num, width)))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            config=config,
            attr_count=config.attr_count,
            visual_or_text=True,
            prompt_num=prompt_num,
            part_num=self.part_num,
            row_patch_num=row_patch_num,
        )
        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x],
            dim=1,
        )
        x = x + self.positional_embedding.to(x.dtype)
        if hasattr(self, "part_class_embedding"):
            x = torch.cat(
                [
                    x[:, :1],
                    self.part_class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                    x[:, 1:],
                ],
                dim=1,
            )
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x, all_class, attnmap = self.transformer(x)
        x = x.permute(1, 0, 2)
        if self.proj is not None:
            x = x @ self.proj
            if all_class is not None:
                all_class = all_class @ self.proj
        return x, all_class, attnmap


class SoftmaxWithTemperature(nn.Module):
    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(initial_temperature))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits / self.temperature, dim=-1)


class CLIP(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        image_resolution: int,
        vision_layers: int,
        vision_width: int,
        vision_patch_size: int,
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        config: PromptPARConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__()
        self.config = config
        self.context_length = context_length
        self.vis_prompt_len = config.vis_prompt
        self.text_prompt_len = config.text_prompt if config.use_textprompt else 0
        if config.use_gl:
            self.softmax_model = SoftmaxWithTemperature()
            self.agg_bn = nn.BatchNorm1d(config.attr_count)

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            width=vision_width,
            layers=vision_layers,
            heads=vision_width // 64,
            output_dim=embed_dim,
            prompt_num=self.vis_prompt_len,
            config=config,
        )
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            config=config,
            attr_count=config.attr_count,
            attn_mask=self.build_attention_mask(),
            prompt_num=self.text_prompt_len,
        )
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_parameters()

    def initialize_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self) -> torch.Tensor:
        mask = torch.empty(self.context_length + self.text_prompt_len, self.context_length + self.text_prompt_len)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype

    def encode_text(self, text: torch.Tensor) -> torch.Tensor:
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        return x[torch.arange(x.shape[0], device=text.device), text.argmax(dim=-1)] @ self.text_projection

    def forward_aggregate(self, image: torch.Tensor, text: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        all_class = (image / image.norm(dim=-1, keepdim=True)).float()
        text_features = (text / text.norm(dim=-1, keepdim=True)).float()
        logits_per_image = self.logit_scale.exp() * all_class @ text_features.t()
        similarity = self.softmax_model(logits_per_image)
        global_similarity = similarity[:, 0]
        local_similarity = similarity[:, 1:]
        similarity_aggregate = global_similarity
        for logits_local in local_similarity:
            max_values, _ = torch.max(logits_local, dim=0)
            min_values, _ = torch.min(logits_local, dim=0)
            gamma = max_values > self.config.default_threshold
            similarity_aggregate = gamma.float() * max_values + (1 - gamma.float()) * min_values
        final_similarity = (similarity_aggregate + global_similarity) / 2
        return self.agg_bn(final_similarity), logits_per_image


def convert_weights(model: nn.Module) -> None:
    def _convert_weights_to_fp16(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            module.weight.data = module.weight.data.half()
            if module.bias is not None:
                module.bias.data = module.bias.data.half()
        if isinstance(module, nn.MultiheadAttention):
            for attr in [*[f"{prefix}_proj_weight" for prefix in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(module, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()
        for name in ["text_projection", "proj"]:
            if hasattr(module, name):
                tensor = getattr(module, name)
                if tensor is not None:
                    tensor.data = tensor.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict[str, torch.Tensor], config: PromptPARConfig = DEFAULT_CONFIG) -> CLIP:
    if "visual.proj" not in state_dict:
        raise ValueError("This standalone package only supports ViT-based PromptPAR checkpoints.")

    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len([key for key in state_dict if key.startswith("visual.") and key.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(key.split(".")[2] for key in state_dict if key.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim=embed_dim,
        image_resolution=image_resolution,
        vision_layers=vision_layers,
        vision_width=vision_width,
        vision_patch_size=vision_patch_size,
        context_length=context_length,
        vocab_size=vocab_size,
        transformer_width=transformer_width,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers,
        config=config,
    )
    state_dict = dict(state_dict)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)
    convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    return model.eval()
