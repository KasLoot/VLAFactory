# Portions adapted from Hugging Face Transformers v5.1.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LFM2 hybrid backbone. Adapted from Transformers v5.1.0 (Apache-2.0)."""

import torch
from torch import nn
from torch.nn import functional as F
from .config import TextConfig
from .cache import ModelCache


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + self.eps)
        return self.weight * y.to(x.dtype)


def rotary_embeddings(positions, head_dim, theta, dtype):
    # Compute frequencies in fp32 even when the enclosing model is in BF16.
    with torch.autocast(device_type=positions.device.type, enabled=False):
        inv = 1.0 / theta ** (
            torch.arange(0, head_dim, 2, device=positions.device).float() / head_dim
        )
        freqs = positions.float().unsqueeze(-1) * inv
        freqs = torch.cat((freqs, freqs), dim=-1)
        return freqs.cos().to(dtype).unsqueeze(1), freqs.sin().to(dtype).unsqueeze(1)


def apply_rotary(x, cos, sin):
    a, b = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-b, a), dim=-1) * sin


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        self.q_proj = nn.Linear(
            config.hidden_size, self.heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.kv_heads * self.head_dim, bias=False
        )
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, config.norm_eps)

    def forward(self, x, rotary, mask, cache, index):
        b, s, _ = x.shape
        q = self.q_layernorm(
            self.q_proj(x).view(b, s, self.heads, self.head_dim)
        ).transpose(1, 2)
        k = self.k_layernorm(
            self.k_proj(x).view(b, s, self.kv_heads, self.head_dim)
        ).transpose(1, 2)
        v = self.v_proj(x).view(b, s, self.kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary(q, *rotary), apply_rotary(k, *rotary)
        if cache is not None:
            if index in cache.attention:
                old_k, old_v = cache.attention[index]
                k, v = torch.cat((old_k, k), dim=2), torch.cat((old_v, v), dim=2)
            cache.attention[index] = (k, v)
        # GQA with an explicit mask can select a different kernel. Use the same
        # layout as the reference: native GQA for unmasked prefill/decoding and
        # repeated KV heads for padded batches or chunked continuation.
        use_gqa = mask is None
        if not use_gqa:
            groups = self.heads // self.kv_heads
            k = k.repeat_interleave(groups, dim=1)
            v = v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=mask is None and s > 1,
            enable_gqa=use_gqa,
        )
        return self.out_proj(y.transpose(1, 2).reshape(b, s, -1))


class ShortConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_size
        self.kernel_size = config.conv_kernel_size
        self.in_proj = nn.Linear(d, 3 * d, bias=config.conv_bias)
        self.conv = nn.Conv1d(d, d, self.kernel_size, groups=d, bias=config.conv_bias)
        self.out_proj = nn.Linear(d, d, bias=config.conv_bias)

    def forward(self, x, valid, cache, index):
        x = x * valid.unsqueeze(-1).to(x.dtype)
        b, c, x = self.in_proj(x).transpose(1, 2).chunk(3, dim=1)
        bx = b * x
        history = None if cache is None else cache.convolution.get(index)
        if history is None:
            # Symmetric padding plus truncation matches the reference prefill
            # kernel while implementing a strictly causal convolution.
            conv_out = F.conv1d(
                bx,
                self.conv.weight,
                self.conv.bias,
                padding=self.kernel_size - 1,
                groups=bx.shape[1],
            )[..., : bx.shape[-1]]
            combined = F.pad(bx, (self.kernel_size - 1, 0))
        else:
            combined = torch.cat((history, bx), dim=-1)
            if bx.shape[-1] == 1:
                conv_out = (combined * self.conv.weight[:, 0, :]).sum(-1, keepdim=True)
                if self.conv.bias is not None:
                    conv_out = conv_out + self.conv.bias[None, :, None]
            else:
                conv_out = self.conv(combined)
        if cache is not None:
            cache.convolution[index] = (
                combined[..., -(self.kernel_size - 1) :].clone()
                if self.kernel_size > 1
                else combined[..., :0]
            )
        y = c * conv_out
        return self.out_proj(y.transpose(1, 2))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class DecoderLayer(nn.Module):
    def __init__(self, config, kind):
        super().__init__()
        self.kind = kind
        if kind == "full_attention":
            self.self_attn = Attention(config)
        else:
            self.conv = ShortConv(config)
        self.feed_forward = FeedForward(config)
        self.operator_norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(self, x, rotary, mask, valid, cache, index):
        normalized = self.operator_norm(x)
        if self.kind == "full_attention":
            x = x + self.self_attn(normalized, rotary, mask, cache, index)
        else:
            x = x + self.conv(normalized, valid, cache, index)
        return x + self.feed_forward(self.ffn_norm(x))


class LanguageModel(nn.Module):
    def __init__(self, config: TextConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            DecoderLayer(config, kind) for kind in config.layer_types
        )
        self.embedding_norm = RMSNorm(config.hidden_size, config.norm_eps)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
        position_ids=None,
        cache: ModelCache | None = None,
    ):
        x = inputs_embeds
        batch, length, _ = x.shape
        past = 0 if cache is None else cache.length
        if length == 0:
            raise ValueError("Input must contain at least one token")
        if attention_mask is None:
            previous = None if cache is None else cache.attention_mask
            current = torch.ones(batch, length, device=x.device, dtype=torch.bool)
            attention_mask = (
                current if previous is None else torch.cat((previous, current), dim=1)
            )
        attention_mask = attention_mask.to(device=x.device, dtype=torch.bool)
        if attention_mask.shape != (batch, past + length):
            raise ValueError("attention_mask must cover both cached and current tokens")
        valid = attention_mask[:, -length:]
        if position_ids is None:
            position_ids = (attention_mask.long().cumsum(-1) - 1).clamp_min(0)[
                :, -length:
            ]
        if position_ids.max() >= self.config.max_position_embeddings:
            raise ValueError("Position exceeds configured limit")
        q_pos = torch.arange(past, past + length, device=x.device)
        k_pos = torch.arange(past + length, device=x.device)
        mask = (k_pos[None, :] <= q_pos[:, None])[None, None] & attention_mask[
            :, None, None, :
        ]
        if attention_mask.all() and (past == 0 or length == 1):
            mask = None
        rotary = rotary_embeddings(
            position_ids,
            self.config.hidden_size // self.config.num_attention_heads,
            self.config.rope_theta,
            x.dtype,
        )
        for i, layer in enumerate(self.layers):
            x = layer(x, rotary, mask, valid, cache, i)
        if cache is not None:
            cache.length += length
            cache.attention_mask = attention_mask
        return self.embedding_norm(x)
