"""Copy and label DiffusionDB images without changing their pixels or dimensions."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
import pyarrow.dataset as ds
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
METADATA_COLUMNS = [
    "part_id", "image_name", "width", "height", "image_nsfw", "prompt_nsfw",
]


def _load_metadata(parquet_path, part_ids):
    """Read only the extra attributes for the downloaded parts, keyed by source."""
    metadata = {}
    scanner = ds.dataset(parquet_path, format="parquet").scanner(
        columns=METADATA_COLUMNS,
        filter=ds.field("part_id").isin(part_ids),
    )
    for batch in scanner.to_batches():
        for row in batch.to_pylist():
            key = (row["part_id"], row["image_name"])
            if key in metadata:
                raise ValueError(f"Duplicate Parquet image key: {key!r}")
            metadata[key] = {
                name: row[name]
                for name in ("width", "height", "image_nsfw", "prompt_nsfw")
            }
    return metadata


def _is_finite_number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _valid_labels(labels):
    if not isinstance(labels, dict):
        return False
    return (
        isinstance(labels.get("p"), str) and bool(labels["p"].strip())
        and isinstance(labels.get("sa"), str) and bool(labels["sa"].strip())
        and type(labels.get("se")) is int and 0 <= labels["se"] <= 2**32 - 1
        and type(labels.get("st")) is int and labels["st"] >= 0
        and _is_finite_number(labels.get("c"))
    )


def _valid_metadata(metadata):
    return (
        all(type(metadata[key]) is int and metadata[key] > 0
            for key in ("width", "height"))
        and _is_finite_number(metadata["image_nsfw"])
        and (0 <= metadata["image_nsfw"] <= 1 or metadata["image_nsfw"] == 2)
        and _is_finite_number(metadata["prompt_nsfw"])
        and 0 <= metadata["prompt_nsfw"] <= 1
    )


def _replace_output(staging_dir, output_dir):
    """Publish a completed run; restore the old output if publication fails."""
    backup_dir = None
    if output_dir.exists():
        backup_dir = output_dir.with_name(f".{output_dir.name}.previous-{uuid4().hex}")
        output_dir.rename(backup_dir)
    try:
        staging_dir.rename(output_dir)
    except BaseException:
        if backup_dir is not None:
            backup_dir.rename(output_dir)
        raise
    if backup_dir is not None:
        shutil.rmtree(backup_dir)


def process_diffusiondb_data(workspace_dir, post_processed_dataset_name):
    """Build a fresh processed dataset from poloclub/images and metadata.parquet.

    Keep source images byte-for-byte, with their original resolutions and aspect
    ratios. Exclude invalid records, unreadable images, completely black images,
    and images marked as censored (image_nsfw == 2.0). RGB decoding is only used
    for validation; no conversion, resizing, normalization, or batching is saved.

    Missing/malformed part JSON, missing Parquet matches, and duplicate Parquet
    keys abort the run. Existing output is replaced only after a successful build.
    The output dataset name may be a relative path such as VLAFactory/val.
    Return counts of copied and skipped images along with the output directory.
    """
    dataset_path = Path(post_processed_dataset_name)
    if (not post_processed_dataset_name
            or not dataset_path.parts
            or dataset_path.is_absolute()
            or ".." in dataset_path.parts):
        raise ValueError("post_processed_dataset_name must be a relative dataset path without '..'")

    datasets_dir = Path(workspace_dir) / "data" / "dataset"
    dataset_dir = datasets_dir / "poloclub" / "diffusiondb"
    images_dir = dataset_dir / "images/train"
    output_dir = datasets_dir / dataset_path / "post_processed_images"
    if not output_dir.resolve().is_relative_to(datasets_dir.resolve()):
        raise ValueError("Output must remain inside the datasets directory")

    part_dirs = sorted(path for path in images_dir.glob("part-*") if path.is_dir())
    if not part_dirs:
        raise FileNotFoundError(f"No part directories found in {images_dir}")
    if any(not path.name.removeprefix("part-").isdigit() for path in part_dirs):
        raise ValueError("Expected part directory names of the form part-000001")
    part_ids = [int(path.name.removeprefix("part-")) for path in part_dirs]
    if len(set(part_ids)) != len(part_ids):
        raise ValueError("Multiple part directories have the same numeric part ID")
    for part_dir in part_dirs:
        json_path = part_dir / f"{part_dir.name}.json"
        if not json_path.is_file():
            raise FileNotFoundError(f"Missing part metadata: {json_path}")

    metadata = _load_metadata(dataset_dir / "metadata.parquet", part_ids)
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        raise ValueError(f"Output must be a regular directory: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    full_json = []
    skipped = Counter()
    with tempfile.TemporaryDirectory(
        prefix=".post_processed_images.build-", dir=output_dir.parent,
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        for part_dir, part_id in zip(part_dirs, part_ids):
            json_path = part_dir / f"{part_dir.name}.json"
            with json_path.open("r", encoding="utf-8") as file:
                part_json = json.load(file)
            if not isinstance(part_json, dict):
                raise ValueError(f"Expected filename-keyed JSON object: {json_path}")

            image_paths = sorted(
                path for path in part_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            for image_path in tqdm(image_paths, desc=f"Processing {part_dir.name}"):
                key = (part_id, image_path.name)
                if key not in metadata:
                    raise ValueError(f"Missing Parquet metadata for {key!r}")
                extra = metadata.pop(key)
                labels = part_json.get(image_path.name)
                if not _valid_labels(labels):
                    skipped["invalid_labels"] += 1
                    continue
                if not _valid_metadata(extra):
                    skipped["invalid_metadata"] += 1
                    continue
                if extra["image_nsfw"] == 2.0:
                    skipped["censored"] += 1
                    continue

                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        image.load()
                        dimensions = image.size
                        # Inspect RGB values without rewriting the source image.
                        with image.convert("RGB") as rgb:
                            is_black = all(maximum == 0 for _, maximum in rgb.getextrema())
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    skipped["unreadable_image"] += 1
                    continue
                if dimensions != (extra["width"], extra["height"]):
                    skipped["dimension_mismatch"] += 1
                    continue
                if is_black:
                    skipped["blackout"] += 1
                    continue

                image_name = f"{len(full_json)}{image_path.suffix.lower()}"
                shutil.copy2(image_path, staging_dir / image_name)
                full_json.append({
                    "image": image_name,
                    "prompt": labels["p"],
                    "se": labels["se"],
                    "c": labels["c"],
                    "st": labels["st"],
                    "sa": labels["sa"],
                    "source_image_name": image_path.name,
                    "source_part_id": part_id,
                    **extra,
                })

        json_path = staging_dir / "post_processed_images-labels.json"
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(full_json, file, ensure_ascii=False, indent=4, allow_nan=False)
            file.write("\n")
        _replace_output(staging_dir, output_dir)

    return {
        "output_dir": str(output_dir),
        "copied": len(full_json),
        "skipped": sum(skipped.values()),
        "skipped_by_reason": dict(sorted(skipped.items())),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-dir", default="/home/yuxin/workspace")
    parser.add_argument(
        "--dataset-name", default="VLAFactory/train",
        help="Output dataset path relative to data/datasets (e.g. VLAFactory/val)",
    )
    args = parser.parse_args()
    summary = process_diffusiondb_data(args.workspace_dir, args.dataset_name)
    print(json.dumps(summary, indent=2))
