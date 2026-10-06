# Portions adapted from Hugging Face Transformers v5.1.0.
# Copyright 2025 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Image preprocessing and ChatML formatting, independent of Transformers.

Image arithmetic and tile ordering follow Transformers v5.1.0 (Apache-2.0).
"""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms.v2 import functional as tvf
from .config import ImageConfig
from .tokenizer import LFMTokenizer


class Batch(dict):
    def to(self, device, dtype=None):
        return Batch(
            {
                key: value.to(
                    device=device, dtype=dtype if value.is_floating_point() else None
                )
                for key, value in self.items()
            }
        )


def format_chat(
    messages,
    *,
    add_generation_prompt=True,
    continue_final_message=False,
    tools=None,
    keep_past_thinking=False,
    date=None,
):
    """Render the released chat template, including text/image and tool messages."""
    if not messages:
        raise ValueError("messages cannot be empty")
    if add_generation_prompt and continue_final_message:
        raise ValueError("Cannot both continue a message and add an assistant prompt")

    def content(value):
        if isinstance(value, str):
            return value
        return "".join(
            "<image>"
            if item["type"] == "image"
            else item["text"]
            if item["type"] == "text"
            else json.dumps(item, ensure_ascii=False)
            for item in value
        )

    def arg(value):
        if isinstance(value, str):
            return '"' + value + '"'
        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    messages = list(messages)
    system = ""
    if messages[0]["role"] == "system":
        system = content(messages.pop(0).get("content", ""))
    if tools:
        date = date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        system += (
            ("\n\n" if system else "")
            + "Today's date: "
            + date
            + "\n\nList of tools: "
            + json.dumps(tools, ensure_ascii=False)
        )
    parts = ["<|startoftext|>"]
    if system:
        parts.append("<|im_start|>system\n" + system + "<|im_end|>\n")
    last_assistant = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"), default=-1
    )
    for i, message in enumerate(messages):
        role = message["role"]
        parts.append("<|im_start|>" + role + "\n")
        if role == "assistant":
            if "thinking" in message and (keep_past_thinking or i == last_assistant):
                parts.append("<think>" + message["thinking"] + "</think>")
            if "tool_calls" in message:
                calls = []
                for call in message["tool_calls"]:
                    fn = call["function"]
                    args = ", ".join(
                        k + "=" + arg(v) for k, v in fn["arguments"].items()
                    )
                    calls.append(fn["name"] + "(" + args + ")")
                parts.append(
                    "<|tool_call_start|>[" + ", ".join(calls) + "]<|tool_call_end|>"
                )
        if "content" in message:
            text = content(message["content"])
            if (
                role == "assistant"
                and not keep_past_thinking
                and i != last_assistant
                and "</think>" in text
            ):
                text = text.split("</think>")[-1].strip()
            parts.append(text)
            if not (
                role == "assistant"
                and continue_final_message
                and i == len(messages) - 1
            ):
                parts.append("<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


class ImageProcessor:
    def __init__(self, config: ImageConfig | None = None):
        self.config = config or ImageConfig()

    def smart_resize(self, height, width):
        c = self.config
        factor = c.encoder_patch_size * c.downsample_factor
        minimum, maximum = (
            c.min_image_tokens * factor**2,
            c.max_image_tokens * factor**2,
        )
        h, w = (
            max(factor, round(height / factor) * factor),
            max(factor, round(width / factor) * factor),
        )
        if h * w > maximum:
            beta = math.sqrt(height * width / maximum)
            h, w = (
                max(factor, math.floor(height / beta / factor) * factor),
                max(factor, math.floor(width / beta / factor) * factor),
            )
        elif h * w < minimum:
            beta = math.sqrt(minimum / (height * width))
            h, w = (
                math.ceil(height * beta / factor) * factor,
                math.ceil(width * beta / factor) * factor,
            )
        return h, w

    def _tiles(self, image):
        c = self.config
        h, w = image.shape[-2:]
        small_h, small_w = self.smart_resize(h, w)
        factor = c.encoder_patch_size * c.downsample_factor
        area = max(c.encoder_patch_size, round(h / factor) * factor) * max(
            c.encoder_patch_size, round(w / factor) * factor
        )
        split = c.do_image_splitting and not c.min_tiles == c.max_tiles == 1
        if split and area > c.max_image_tokens * factor**2 * c.max_pixels_tolerance:
            ratios = sorted(
                {
                    (x, y)
                    for n in range(c.min_tiles, c.max_tiles + 1)
                    for x in range(1, n + 1)
                    for y in range(1, n + 1)
                    if c.min_tiles <= x * y <= c.max_tiles
                },
                key=lambda r: r[0] * r[1],
            )
            best, difference = (1, 1), float("inf")
            for ratio in ratios:
                diff = abs(w / h - ratio[0] / ratio[1])
                if diff < difference or (
                    diff == difference
                    and w * h > 0.5 * c.tile_size**2 * ratio[0] * ratio[1]
                ):
                    best, difference = ratio, diff
            cols, rows = best
            large = tvf.resize(
                image,
                [rows * c.tile_size, cols * c.tile_size],
                interpolation=tvf.InterpolationMode.BILINEAR,
                antialias=True,
            )
            tiles = [
                large[
                    :,
                    r * c.tile_size : (r + 1) * c.tile_size,
                    k * c.tile_size : (k + 1) * c.tile_size,
                ]
                for r in range(rows)
                for k in range(cols)
            ]
            if c.use_thumbnail:
                tiles.append(
                    tvf.resize(
                        image,
                        [small_h, small_w],
                        interpolation=tvf.InterpolationMode.BILINEAR,
                        antialias=True,
                    )
                )
        else:
            rows = cols = 1
            tiles = [
                tvf.resize(
                    image,
                    [small_h, small_w],
                    interpolation=tvf.InterpolationMode.BILINEAR,
                    antialias=True,
                )
            ]
        return tiles, rows, cols, (small_h, small_w)

    def __call__(self, images):
        """Process a flat image list. Inputs are PIL RGB images or uint8 CHW tensors."""
        c = self.config
        p = c.encoder_patch_size
        maximum = max(
            c.max_image_tokens * c.downsample_factor**2,
            (c.tile_size // p) ** 2 if c.do_image_splitting else 0,
        )
        patches, masks, shapes, metadata = [], [], [], []
        for image in images:
            if isinstance(image, (str, Path)):
                with Image.open(image) as opened:
                    image = opened.convert("RGB")
            if isinstance(image, Image.Image):
                image = tvf.pil_to_tensor(image.convert("RGB"))
            if (
                not isinstance(image, torch.Tensor)
                or image.ndim != 3
                or image.shape[0] != 3
                or image.dtype != torch.uint8
            ):
                raise ValueError(
                    "Expected a PIL image, local image path, or uint8 RGB CHW tensor"
                )
            if min(image.shape[-2:]) < 1:
                raise ValueError("Empty image")
            tiles, rows, cols, size = self._tiles(image)
            metadata.append((rows, cols, size))
            for tile in tiles:
                h, w = tile.shape[-2:]
                # Match the reference fused normalization: (uint8 - 127.5) / 127.5.
                tile = (tile.float() - 127.5) / 127.5
                grid_h, grid_w = h // p, w // p
                x = (
                    tile.reshape(3, grid_h, p, grid_w, p)
                    .permute(1, 3, 2, 4, 0)
                    .reshape(grid_h * grid_w, -1)
                )
                if len(x) > maximum:
                    raise ValueError(
                        "Aspect ratio exceeds configured patch budget; resize the input or raise max_image_tokens"
                    )
                patches.append(F.pad(x, (0, 0, 0, maximum - len(x))))
                masks.append(torch.arange(maximum, device=x.device) < len(x))
                shapes.append((grid_h, grid_w))
        if not patches:
            raise ValueError("No images supplied")
        return Batch(
            pixel_values=torch.stack(patches),
            pixel_attention_mask=torch.stack(masks),
            spatial_shapes=torch.tensor(shapes, device=patches[0].device),
        ), metadata

    def image_tokens(self, metadata):
        c = self.config
        rows, cols, (h, w) = metadata
        factor = c.encoder_patch_size * c.downsample_factor
        parts = ["<|image_start|>"] if c.use_image_special_tokens else []
        if rows > 1 or cols > 1:
            for row in range(rows):
                for col in range(cols):
                    if c.use_image_special_tokens:
                        parts.append(f"<|img_row_{row + 1}_col_{col + 1}|>")
                    parts.append("<image>" * (c.tile_size // factor) ** 2)
            if c.use_thumbnail:
                if c.use_image_special_tokens:
                    parts.append("<|img_thumbnail|>")
                parts.append("<image>" * ((h // factor) * (w // factor)))
        else:
            parts.append("<image>" * ((h // factor) * (w // factor)))
        if c.use_image_special_tokens:
            parts.append("<|image_end|>")
        return "".join(parts)


class Processor:
    def __init__(self, tokenizer=None, image_config=None):
        self.tokenizer = tokenizer or LFMTokenizer.from_pretrained()
        self.image_processor = ImageProcessor(image_config)

    def __call__(
        self, text, images=None, *, add_special_tokens=True, padding_side="left"
    ):
        texts = [text] if isinstance(text, str) else list(text)
        if not texts or padding_side not in ("left", "right"):
            raise ValueError("Provide nonempty texts and left/right padding_side")
        # Single text takes a flat list; a text batch takes a nested list.
        nested = (
            [[] for _ in texts]
            if images is None
            else ([list(images)] if isinstance(text, str) else list(images))
        )
        if len(nested) != len(texts) or any(
            t.count("<image>") != len(imgs) for t, imgs in zip(texts, nested)
        ):
            raise ValueError(
                "Each prompt must have one <image> placeholder per supplied image"
            )
        result = Batch()
        flat_images = [image for group in nested for image in group]
        if flat_images:
            result, metadata = self.image_processor(flat_images)
            metadata = iter(metadata)
            expanded = []
            for text in texts:
                parts = text.split("<image>")
                expanded.append(
                    "".join(
                        part + self.image_processor.image_tokens(next(metadata))
                        for part in parts[:-1]
                    )
                    + parts[-1]
                )
            texts = expanded
        encoded = self.tokenizer.encode_batch(
            texts, add_special_tokens=add_special_tokens
        )
        maximum = max(map(len, encoded))
        ids = torch.full(
            (len(encoded), maximum), self.tokenizer.pad_token_id, dtype=torch.long
        )
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, tokens in enumerate(encoded):
            start = maximum - len(tokens) if padding_side == "left" else 0
            ids[i, start : start + len(tokens)] = torch.tensor(tokens, dtype=torch.long)
            mask[i, start : start + len(tokens)] = True
        result.update(input_ids=ids, attention_mask=mask)
        return result

    def apply_chat_template(self, messages, *, tokenize=True, **kwargs):
        prompt = format_chat(messages, **kwargs)
        if not tokenize:
            return prompt
        images = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if item["type"] == "image":
                        if "image" not in item:
                            raise ValueError(
                                "Image content requires an image object or local path in the image field"
                            )
                        images.append(item["image"])
        # Chat template already includes BOS.
        return self(prompt, images, add_special_tokens=False)
