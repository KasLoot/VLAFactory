"""Independent freezing and LoRA utilities for architecture experiments."""

import math
import torch
from torch import nn
from torch.nn import functional as F


def set_trainable(module, trainable=True):
    module.requires_grad_(trainable)
    return module


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank=8, alpha=16.0, dropout=0.0):
        super().__init__()
        if rank < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA rank or dropout")
        self.base = base
        self.base.requires_grad_(False)
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        return (
            self.base(x)
            + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        )

    @torch.no_grad()
    def merged_linear(self):
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        merged.weight.copy_(
            self.base.weight
            + (self.lora_B.float() @ self.lora_A.float()).to(self.base.weight.dtype)
            * self.scale
        )
        if merged.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


def inject_lora(
    module,
    *,
    target_names=("q_proj", "k_proj", "v_proj", "out_proj"),
    rank=8,
    alpha=16.0,
    dropout=0.0,
):
    """Replace matching Linear children in the supplied subtree. Freeze separately.

    Names may be local names or full paths relative to module. Returns changed paths.
    Existing LoRA wrappers are not traversed, so repeated calls cannot nest adapters.
    """
    changed = []

    def visit(parent, prefix=""):
        for name, child in list(parent.named_children()):
            path = f"{prefix}.{name}" if prefix else name
            if isinstance(child, LoRALinear):
                continue
            if isinstance(child, nn.Linear) and (
                name in target_names or path in target_names
            ):
                setattr(parent, name, LoRALinear(child, rank, alpha, dropout))
                changed.append(path)
            else:
                visit(child, path)

    visit(module)
    if not changed:
        raise ValueError("No linear layers matched LoRA targets")
    return changed


def trainable_parameters(module):
    return {name: p for name, p in module.named_parameters() if p.requires_grad}
