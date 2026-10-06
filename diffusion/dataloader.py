"""Batch VAE images by exact resolution without resizing or padding.

Usage::

    loader = create_vae_dataloader("/path/to/train", shuffle=True)
    val_loader = create_vae_dataloader("/path/to/val", shuffle=False)
    for epoch in range(num_epochs):
        loader.batch_sampler.set_epoch(epoch)
        for images in loader:
            # images: float32 [B, 3, H, W], normalized by VAEDataset.
            ...

The sampler owns shuffling and batch sizes. Each selected image appears once
per epoch, including samples in incomplete batches and singleton groups.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import random

import torch
from torch.utils.data import DataLoader, Sampler

from diffusion.dataset import VAEDataset


def _integer(name, value, minimum):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class ResolutionBatchSampler(Sampler[list[int]]):
    """Shuffle resolution groups into batches, then shuffle the batch order.

    ``records`` contains the manifest's width/height attributes. ``indices``
    selects records from the original dataset, for example a training split;
    indices are not renumbered. No image decoding occurs in the sampler.

    Batch size is capped by both max_batch_size and max_batch_pixels / (W * H).
    Images larger than the pixel budget are emitted alone. The pixel budget
    controls batch construction; actual GPU memory usage must be profiled.
    """

    def __init__(
        self,
        records: Sequence[Mapping],
        *,
        indices: Iterable[int] | None = None,
        max_batch_pixels: int = 4_194_304,
        max_batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.max_batch_pixels = _integer("max_batch_pixels", max_batch_pixels, 1)
        self.max_batch_size = _integer("max_batch_size", max_batch_size, 1)
        self.seed = _integer("seed", seed, 0)
        self.shuffle = shuffle
        self.epoch = 0

        selected = range(len(records)) if indices is None else indices
        groups = defaultdict(list)
        seen = set()
        for index in selected:
            _integer("dataset index", index, 0)
            if index >= len(records):
                raise ValueError(f"Dataset index {index} is out of range")
            if index in seen:
                raise ValueError(f"Duplicate dataset index: {index}")
            seen.add(index)
            record = records[index]
            try:
                width = _integer("width", record["width"], 1)
                height = _integer("height", record["height"], 1)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid resolution for dataset index {index}") from error
            groups[(width, height)].append(index)

        self.resolution_groups = {
            resolution: tuple(groups[resolution]) for resolution in sorted(groups)
        }

    def batch_size_for_resolution(self, width: int, height: int) -> int:
        _integer("width", width, 1)
        _integer("height", height, 1)
        return min(
            self.max_batch_size,
            max(1, self.max_batch_pixels // (width * height)),
        )

    def set_epoch(self, epoch: int) -> None:
        """Use seed + epoch for reproducible reshuffling on each epoch."""
        self.epoch = _integer("epoch", epoch, 0)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        batches = []
        for (width, height), indices in self.resolution_groups.items():
            group = list(indices)
            if self.shuffle:
                rng.shuffle(group)
            batch_size = self.batch_size_for_resolution(width, height)
            batches.extend(
                group[start:start + batch_size]
                for start in range(0, len(group), batch_size)
            )
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for (width, height), indices in self.resolution_groups.items():
            batch_size = self.batch_size_for_resolution(width, height)
            total += (len(indices) + batch_size - 1) // batch_size
        return total


def create_vae_dataloader(
    dataset_path: str | Path | None = None,
    *,
    indices: Iterable[int] | None = None,
    max_batch_pixels: int = 4_194_304,
    max_batch_size: int = 32,
    shuffle: bool = True,
    seed: int = 0,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a loader for an explicitly provided split directory.

    ``dataset_path`` must contain the images and their manifest directly. Missing
    or empty paths raise ValueError. Parent directories are not expanded into
    train/val paths; the caller chooses the directory and shuffling behavior.
    Use shuffle=False for validation. ``indices`` optionally selects a subset of
    this split. Call batch_sampler.set_epoch(epoch) before each training epoch.

    RGB conversion/normalization belong to the dataset; the loader uses default
    tensor stacking for matching resolutions.
    """
    dataset = VAEDataset(dataset_path)
    _integer("num_workers", num_workers, 0)
    _integer("prefetch_factor", prefetch_factor, 1)
    if len(dataset.dataset_json) != len(dataset):
        raise ValueError("Dataset length must match its manifest record count")

    sampler = ResolutionBatchSampler(
        dataset.dataset_json,
        indices=indices,
        max_batch_pixels=max_batch_pixels,
        max_batch_size=max_batch_size,
        shuffle=shuffle,
        seed=seed,
    )
    # Isolate the DataLoader worker seeds from the global torch RNG.
    generator = torch.Generator().manual_seed(seed)
    worker_options = {}
    if num_workers > 0:
        worker_options.update(
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
        **worker_options,
    )
