# Portions adapted from Hugging Face Transformers v5.1.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Trainable LFM2.5-VL assembly with direct feature access and tied output weights."""

from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from .config import ModelConfig
from .cache import ModelCache
from .language import LanguageModel
from .vision import VisionTower


class MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.factor = config.downsample_factor
        width = config.vision.hidden_size * self.factor**2
        self.layer_norm = (
            nn.LayerNorm(width) if config.projector_use_layernorm else None
        )
        self.linear_1 = nn.Linear(
            width, config.projector_hidden_size, bias=config.projector_bias
        )
        self.linear_2 = nn.Linear(
            config.projector_hidden_size,
            config.text.hidden_size,
            bias=config.projector_bias,
        )

    def forward(self, x):
        b, h, w, c = x.shape
        f = self.factor
        if h % f or w % f:
            raise ValueError("Patch grid must be divisible by downsample_factor")
        # Keep the exact checkpoint channel ordering; ordinary pixel_unshuffle differs.
        x = x.reshape(b, h, w // f, c * f).permute(0, 2, 1, 3)
        x = x.reshape(b, w // f, h // f, c * f * f).permute(0, 2, 1, 3)
        if self.layer_norm is not None:
            x = self.layer_norm(x)
        return self.linear_2(F.gelu(self.linear_1(x)))


class Backbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_tower = VisionTower(config.vision)
        self.multi_modal_projector = MultiModalProjector(config)
        self.language_model = LanguageModel(config.text)


@dataclass
class ModelOutput:
    logits: torch.Tensor | None
    last_hidden_state: torch.Tensor
    image_features: torch.Tensor | None = None
    cache: ModelCache | None = None
    loss: torch.Tensor | None = None


class LFMModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.model = Backbone(self.config)

    @property
    def output_weight(self):
        """The LM head always shares the input embedding parameter."""
        return self.model.language_model.embed_tokens.weight

    def encode_images(self, pixel_values, spatial_shapes, pixel_attention_mask):
        weight = self.model.vision_tower.vision_model.embeddings.patch_embedding.weight
        pixels = pixel_values.to(device=weight.device, dtype=weight.dtype)
        shapes = spatial_shapes.to(weight.device)
        mask = pixel_attention_mask.to(weight.device)
        states = self.model.vision_tower(pixels, shapes, mask)
        features = []
        for state, (h, w), valid in zip(states, shapes.tolist(), mask):
            if (
                int(valid.sum()) != h * w
                or not valid[: h * w].all()
                or valid[h * w :].any()
            ):
                raise ValueError("Vision masks must describe contiguous valid patches")
            grid = state[: h * w].reshape(1, h, w, -1)
            features.append(
                self.model.multi_modal_projector(grid).reshape(
                    -1, self.config.text.hidden_size
                )
            )
        return torch.cat(features, dim=0)

    def forward(
        self,
        input_ids=None,
        *,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        spatial_shapes=None,
        pixel_attention_mask=None,
        cache=None,
        use_cache=False,
        labels=None,
        logits_to_keep=0,
        return_logits=True,
    ):
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Provide exactly one of input_ids and inputs_embeds")
        if pixel_values is not None and (
            input_ids is None or spatial_shapes is None or pixel_attention_mask is None
        ):
            raise ValueError(
                "Images require input_ids, spatial_shapes, and pixel_attention_mask"
            )
        if labels is not None and (logits_to_keep or not return_logits):
            raise ValueError("Training loss requires logits for the full input")
        if logits_to_keep < 0:
            raise ValueError("logits_to_keep cannot be negative")
        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        if use_cache and cache is None:
            cache = ModelCache()
        past = 0 if cache is None else cache.length
        if past + inputs_embeds.shape[1] > self.config.context_length:
            raise ValueError("Sequence exceeds configured context_length")
        image_features = None
        if pixel_values is not None:
            if past:
                raise ValueError(
                    "Supply images only during prefill, not cached continuation"
                )
            image_features = self.encode_images(
                pixel_values, spatial_shapes, pixel_attention_mask
            )
            locations = input_ids == self.config.image_token_id
            if int(locations.sum()) != image_features.shape[0]:
                raise ValueError(
                    "Image placeholders and projected features have different lengths"
                )
            inputs_embeds = inputs_embeds.masked_scatter(
                locations.unsqueeze(-1), image_features.to(inputs_embeds.dtype)
            )
        hidden = self.model.language_model(
            inputs_embeds, attention_mask, position_ids, cache
        )
        logits = (
            F.linear(
                hidden[:, -logits_to_keep:] if logits_to_keep else hidden,
                self.output_weight,
            )
            if return_logits
            else None
        )
        loss = None
        if labels is not None:
            targets = labels[:, 1:].clone()
            if attention_mask is not None:
                supervised = (
                    attention_mask[:, 1:].bool() & attention_mask[:, :-1].bool()
                )
                targets.masked_fill_(~supervised, -100)
            if input_ids is not None:
                targets.masked_fill_(
                    input_ids[:, 1:] == self.config.image_token_id, -100
                )
            if not (targets != -100).any():
                raise ValueError("No supervised next-token targets")
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(logits, hidden, image_features, cache, loss)
