# Portions adapted from Hugging Face Transformers v5.1.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the native LFM2.5-VL architecture (no Transformers dependency)."""

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


def _known(cls, values):
    return {f.name: values[f.name] for f in fields(cls) if f.name in values}


@dataclass
class TextConfig:
    vocab_size: int = 65536
    hidden_size: int = 1024
    intermediate_size: int = 4608  # Effective SwiGLU width, after upstream adjustment.
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    layer_types: tuple[str, ...] = (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    norm_eps: float = 1e-5
    conv_kernel_size: int = 3
    conv_bias: bool = False
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 128000

    def __post_init__(self):
        self.layer_types = tuple(self.layer_types)
        if not self.layer_types or set(self.layer_types) - {"conv", "full_attention"}:
            raise ValueError("layer_types must contain conv or full_attention blocks")
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads % self.num_key_value_heads
        ):
            raise ValueError("Incompatible attention dimensions")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if self.conv_kernel_size < 1:
            raise ValueError("conv_kernel_size must be positive")


@dataclass
class VisionConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    num_channels: int = 3
    num_patches: int = 256
    patch_size: int = 16
    layer_norm_eps: float = 1e-6

    def __post_init__(self):
        if (
            self.hidden_size % self.num_attention_heads
            or int(self.num_patches**0.5) ** 2 != self.num_patches
        ):
            raise ValueError("Invalid vision head dimension or positional grid")


@dataclass
class ImageConfig:
    encoder_patch_size: int = 16
    downsample_factor: int = 2
    min_image_tokens: int = 64
    max_image_tokens: int = 256
    do_image_splitting: bool = True
    min_tiles: int = 2
    max_tiles: int = 10
    tile_size: int = 512
    use_thumbnail: bool = True
    max_pixels_tolerance: float = 2.0
    use_image_special_tokens: bool = True

    def __post_init__(self):
        if not 1 <= self.min_image_tokens <= self.max_image_tokens:
            raise ValueError("Invalid image token limits")
        if not 1 <= self.min_tiles <= self.max_tiles <= 10:
            raise ValueError(
                "Tile limits must be in [1, 10] for the released tokenizer"
            )
        if self.tile_size % (self.encoder_patch_size * self.downsample_factor):
            raise ValueError(
                "tile_size must be divisible by patch_size * downsample_factor"
            )


@dataclass
class ModelConfig:
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    downsample_factor: int = 2
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    projector_use_layernorm: bool = False
    image_token_id: int = 396
    bos_token_id: int = 1
    eos_token_id: int = 7
    pad_token_id: int = 0
    context_length: int = 32768

    @classmethod
    def from_pretrained(cls, directory):
        """Read the original checkpoint config; normalize upstream aliases once."""
        p = Path(directory)
        raw = json.loads((p / "config.json").read_text())
        if "text" in raw:
            return cls.from_dict(raw)
        t = dict(raw["text_config"])
        rope = t.get("rope_parameters", {})
        if rope.get("rope_type", "default") != "default":
            raise ValueError("Only default RoPE is implemented")
        width = t["intermediate_size"]
        if t.get("block_auto_adjust_ff_dim", False):
            width = int(2 * width / 3)
            multiplier = t.get("block_ffn_dim_multiplier")
            if multiplier is not None:
                width = int(multiplier * width)
                multiple = t.get("block_multiple_of", 256)
                width = multiple * ((width + multiple - 1) // multiple)
        t.update(
            intermediate_size=width,
            conv_kernel_size=t.get("conv_L_cache", 3),
            rope_theta=rope.get("rope_theta", t.get("rope_theta", 1e6)),
        )
        v = raw["vision_config"]
        if v.get("hidden_act", "gelu_pytorch_tanh") != "gelu_pytorch_tanh" or v.get(
            "vision_use_head", False
        ):
            raise ValueError("Unsupported vision activation or pooling head")
        if raw.get("projector_hidden_act", "gelu") != "gelu":
            raise ValueError("Only GELU projector is implemented")
        image = dict(raw)
        if (p / "processor_config.json").exists():
            image.update(
                json.loads((p / "processor_config.json").read_text())["image_processor"]
            )
        values = _known(cls, raw)
        values.update(
            text=TextConfig(**_known(TextConfig, t)),
            vision=VisionConfig(**_known(VisionConfig, v)),
            image=ImageConfig(**_known(ImageConfig, image)),
        )
        return cls(**values)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        for key, typ in [
            ("text", TextConfig),
            ("vision", VisionConfig),
            ("image", ImageConfig),
        ]:
            values[key] = typ(**values[key])
        return cls(**values)

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")
