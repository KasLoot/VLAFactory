"""Validated, selective loading from original safetensors; never writes weights."""

from dataclasses import dataclass, field
from fnmatch import fnmatchcase
from pathlib import Path
import torch
from safetensors import safe_open


@dataclass
class LoadReport:
    loaded: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unused: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    incompatible: list[str] = field(default_factory=list)

    def __str__(self):
        return ", ".join(
            f"{name}={len(getattr(self, name))}" for name in self.__dataclass_fields__
        )


class WeightLoadingError(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(
            f"Weight validation failed: {report}\nMissing: {report.missing}\n"
            f"Unused: {report.unused}\nIncompatible: {report.incompatible}"
        )


def _under(name, prefix):
    return name == prefix or name.startswith(prefix.rstrip(".") + ".")


def load_weights(
    model, path, *, include=None, prefix_map=None, allow_missing=(), allow_unused=False
):
    """Load selected target prefixes, validating the complete plan before copying.

    prefix_map maps SOURCE prefixes to TARGET prefixes (or None to omit a source).
    Longest prefixes win. include selects TARGET module prefixes. allow_missing is
    a list of shell-style TARGET parameter patterns for explicitly new parameters.
    Shape mismatches always fail. Load before injecting LoRA or creating optimizers.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "model.safetensors"
    targets = dict(model.named_parameters())
    selected = {
        k: v
        for k, v in targets.items()
        if include is None or any(_under(k, p) for p in include)
    }
    if any(parameter.is_meta for parameter in selected.values()):
        raise ValueError(
            "Instantiate the model on a real device before loading weights"
        )
    if not selected:
        raise ValueError("No target parameters selected")
    mappings = sorted((prefix_map or {}).items(), key=lambda p: len(p[0]), reverse=True)
    report = LoadReport()
    plan = {}
    with safe_open(path, framework="pt", device="cpu") as source:
        for original in source.keys():
            target = original
            for before, after in mappings:
                if _under(original, before):
                    target = None if after is None else after + original[len(before) :]
                    break
            if target is None:
                report.skipped.append(original)
                continue
            if include is not None and not any(_under(target, p) for p in include):
                report.skipped.append(original)
                continue
            if target not in selected:
                report.unused.append(original)
                continue
            if target in plan:
                report.incompatible.append(f"Multiple source tensors map to {target}")
                continue
            shape = tuple(source.get_slice(original).get_shape())
            if shape != tuple(selected[target].shape):
                report.incompatible.append(
                    f"{original} {shape} -> {target} {tuple(selected[target].shape)}"
                )
                continue
            plan[target] = original
        report.missing = sorted(set(selected) - set(plan))
        forbidden_missing = [
            k
            for k in report.missing
            if not any(fnmatchcase(k, p) for p in allow_missing)
        ]
        if (
            report.incompatible
            or forbidden_missing
            or (report.unused and not allow_unused)
        ):
            raise WeightLoadingError(report)
        with torch.no_grad():
            for target, original in plan.items():
                selected[target].copy_(source.get_tensor(original))
                report.loaded.append(target)
    return report
