# Portions adapted from Hugging Face Transformers v5.1.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native SigLIP2 NaFlex encoder. Adapted from Transformers v5.1.0 (Apache-2.0)."""

import torch
from torch import nn
from torch.nn import functional as F
from .config import VisionConfig


class VisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.grid_size = int(config.num_patches**0.5)
        self.patch_embedding = nn.Linear(
            config.num_channels * config.patch_size**2, config.hidden_size
        )
        self.position_embedding = nn.Embedding(config.num_patches, config.hidden_size)

    def forward(self, patches, spatial_shapes):
        x = self.patch_embedding(patches)
        grid = self.position_embedding.weight.reshape(
            self.grid_size, self.grid_size, -1
        ).permute(2, 0, 1)[None]
        source_dtype = grid.dtype
        if grid.device.type == "cpu":
            grid = grid.float()
        positions = []
        for h, w in spatial_shapes.tolist():
            if min(h, w) < 1 or h * w > x.shape[1]:
                raise ValueError("Invalid spatial shape")
            pos = F.interpolate(
                grid, size=(h, w), mode="bilinear", align_corners=False, antialias=True
            )
            pos = pos.flatten(2).transpose(1, 2)[0].to(source_dtype)
            positions.append(torch.cat((pos, pos[:1].expand(x.shape[1] - h * w, -1))))
        return x + torch.stack(positions)


class VisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.heads
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, x, mask):
        b, s, d = x.shape
        q, k, v = [
            proj(x).reshape(b, s, self.heads, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.out_proj(y.transpose(1, 2).reshape(b, s, d))


class VisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x), approximate="tanh"))


class VisionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = VisionAttention(config)
        self.mlp = VisionMLP(config)

    def forward(self, x, mask):
        x = x + self.self_attn(self.layer_norm1(x), mask)
        return x + self.mlp(self.layer_norm2(x))


class VisionEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            VisionLayer(config) for _ in range(config.num_hidden_layers)
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = VisionEmbeddings(config)
        self.encoder = VisionEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(self, pixel_values, spatial_shapes, pixel_attention_mask):
        x = self.embeddings(pixel_values, spatial_shapes)
        mask = pixel_attention_mask[:, None, None, :].bool()
        return self.post_layernorm(self.encoder(x, mask))


class VisionTower(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.vision_model = VisionTransformer(config)

    def forward(self, pixel_values, spatial_shapes, pixel_attention_mask):
        return self.vision_model(pixel_values, spatial_shapes, pixel_attention_mask)
