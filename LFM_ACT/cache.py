"""Explicit per-request state. Create a fresh cache for each independent sequence."""

from dataclasses import dataclass, field
import torch


@dataclass
class ModelCache:
    attention: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    convolution: dict[int, torch.Tensor] = field(default_factory=dict)
    attention_mask: torch.Tensor | None = None
    length: int = 0

    def reset(self):
        self.attention.clear()
        self.convolution.clear()
        self.attention_mask = None
        self.length = 0
