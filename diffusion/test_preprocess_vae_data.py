"""Integration checks for metadata joins, image filtering, and output replacement."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion import preprocess_vae_data as preprocessing


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vae-preprocessing-test-")
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name)
        self.source = self.workspace / "data/dataset/poloclub/diffusiondb"
        self.output = self.workspace / "data/dataset/processed/post_processed_images"
        self.rows = []

    def add_image(self, part_id, name, size=(4, 3), color=(20, 40, 60), **overrides):
        part = self.source / "images/train" / f"part-{part_id:06d}"
        part.mkdir(parents=True, exist_ok=True)
        path = part / name
        with Image.new("RGB", size, color=color) as image:
            image.save(path)
        json_path = part / f"{part.name}.json"
        labels = json.loads(json_path.read_text()) if json_path.exists() else {}
        labels[name] = dict(p=f"prompt for {part_id}/{name}", se=123, c=7.0,
                            st=50, sa="k_euler")
        json_path.write_text(json.dumps(labels), encoding="utf-8")
        self.rows.append({
            "part_id": part_id, "image_name": name,
            "width": size[0], "height": size[1],
            "image_nsfw": 0.25, "prompt_nsfw": 0.1, **overrides,
        })
        return path

    def write_parquet(self):
        schema = pa.schema([
            ("part_id", pa.uint16()), ("image_name", pa.string()),
            ("width", pa.uint16()), ("height", pa.uint16()),
            ("image_nsfw", pa.float32()), ("prompt_nsfw", pa.float32()),
        ])
        pq.write_table(pa.Table.from_pylist(self.rows, schema=schema),
                       self.source / "metadata.parquet")

    def run_preprocessing(self):
        with redirect_stderr(io.StringIO()):
            return preprocessing.process_diffusiondb_data(self.workspace, "processed")

    def labels(self):
        return json.loads((self.output / "post_processed_images-labels.json").read_text())

    def snapshot(self):
        return {path.name: path.read_bytes() for path in self.output.iterdir()}

    def test_nested_dataset_path(self):
        source = self.add_image(1, "a.png")
        self.write_parquet()
        self.output = self.workspace / "data/dataset/VLAFactory/val/post_processed_images"
        with redirect_stderr(io.StringIO()):
            summary = preprocessing.process_diffusiondb_data(self.workspace, "VLAFactory/val")
        self.assertEqual(summary["output_dir"], str(self.output))
        self.assertEqual(summary["copied"], 1)
        self.assertEqual((self.output / "0.png").read_bytes(), source.read_bytes())
        self.assertEqual(self.labels()[0]["source_image_name"], source.name)

    def test_invalid_dataset_paths_are_rejected_before_writing(self):
        for name in ("", ".", "..", "/tmp/processed", "../processed", "VLAFactory/../val"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "relative dataset path"):
                    preprocessing.process_diffusiondb_data(self.workspace, name)
        self.assertFalse((self.workspace / "data").exists())

    def test_dataset_symlink_cannot_escape_datasets_directory(self):
        datasets = self.workspace / "data/dataset"
        datasets.mkdir(parents=True)
        outside = self.workspace / "outside"
        outside.mkdir()
        (datasets / "VLAFactory").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "inside the datasets directory"):
            preprocessing.process_diffusiondb_data(self.workspace, "VLAFactory/val")
        self.assertEqual(list(outside.iterdir()), [])

    def test_join_order_zero_values_and_original_image_bytes(self):
        sources = [
            self.add_image(2, "b.png", size=(9, 4)),
            self.add_image(1, "b.png", size=(7, 3)),
            self.add_image(1, "a.png", size=(3, 5)),
        ]
        part_json = sources[-1].parent / "part-000001.json"
        labels = json.loads(part_json.read_text())
        labels["a.png"].update(se=0, c=0.0, st=0)
        part_json.write_text(json.dumps(labels))
        self.write_parquet()

        summary = self.run_preprocessing()
        records = self.labels()
        self.assertEqual(summary["copied"], 3)
        self.assertEqual(summary["skipped"], 0)
        self.assertEqual([(r["source_part_id"], r["source_image_name"]) for r in records],
                         [(1, "a.png"), (1, "b.png"), (2, "b.png")])
        self.assertEqual((records[0]["se"], records[0]["c"], records[0]["st"]), (0, 0, 0))
        expected_fields = {
            "image", "prompt", "se", "c", "st", "sa", "source_image_name",
            "source_part_id", "width", "height", "image_nsfw", "prompt_nsfw",
        }
        for record, source in zip(records, reversed(sources)):
            self.assertEqual(set(record), expected_fields)
            destination = self.output / record["image"]
            self.assertEqual(destination.read_bytes(), source.read_bytes())
            with Image.open(destination) as image:
                self.assertEqual(image.size, (record["width"], record["height"]))
            self.assertEqual(record["prompt"],
                             f"prompt for {record['source_part_id']}/{source.name}")
            self.assertAlmostEqual(record["image_nsfw"], 0.25)
            self.assertAlmostEqual(record["prompt_nsfw"], 0.1)

    def test_filters_censored_black_corrupt_and_invalid_records(self):
        self.add_image(1, "censored.png", image_nsfw=2.0)
        self.add_image(1, "black.png", color=(0, 0, 0))
        corrupt = self.add_image(1, "corrupt.png")
        corrupt.write_bytes(b"not an image")
        self.add_image(1, "mismatch.png", width=9)
        self.add_image(1, "invalid-metadata.png", prompt_nsfw=float("nan"))
        missing = self.add_image(1, "invalid-labels.png")
        self.add_image(1, "invalid-number.png")
        json_path = missing.parent / "part-000001.json"
        labels = json.loads(json_path.read_text())
        labels[missing.name].pop("se")
        labels["invalid-number.png"]["c"] = 10**400
        json_path.write_text(json.dumps(labels))
        # A very dark real image is retained; no brightness threshold is used.
        self.add_image(1, "dark.png", color=(0, 0, 1), image_nsfw=0.9)
        self.write_parquet()

        summary = self.run_preprocessing()
        self.assertEqual(summary["copied"], 1)
        self.assertEqual(summary["skipped_by_reason"], {
            "censored": 1, "blackout": 1, "unreadable_image": 1,
            "dimension_mismatch": 1, "invalid_metadata": 1, "invalid_labels": 2,
        })
        self.assertEqual(self.labels()[0]["source_image_name"], "dark.png")
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         ["0.png", "post_processed_images-labels.json"])
        self.assertTrue(missing.exists())

    def test_rerun_replaces_all_old_output(self):
        self.add_image(1, "a.png")
        removed = self.add_image(1, "b.png")
        self.write_parquet()
        self.run_preprocessing()
        (self.output / "stale.txt").write_text("old output")
        removed.unlink()
        self.run_preprocessing()
        self.assertEqual(len(self.labels()), 1)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()),
                         ["0.png", "post_processed_images-labels.json"])
        self.assertEqual(list(self.output.parent.glob(".post_processed_images.*")), [])

    def test_missing_first_or_later_part_json_preserves_old_output(self):
        self.add_image(1, "a.png")
        self.write_parquet()
        self.run_preprocessing()
        before = self.snapshot()
        later = self.add_image(2, "a.png")
        self.write_parquet()
        later_json = later.parent / "part-000002.json"
        later_json.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing part metadata"):
            self.run_preprocessing()
        self.assertEqual(before, self.snapshot())
        (self.source / "images/train/part-000001/part-000001.json").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing part metadata"):
            self.run_preprocessing()
        self.assertEqual(before, self.snapshot())

    def test_malformed_later_json_discards_staged_files(self):
        self.add_image(1, "a.png")
        self.write_parquet()
        self.run_preprocessing()
        before = self.snapshot()
        later = self.add_image(2, "a.png")
        (later.parent / "part-000002.json").write_text("not JSON")
        self.write_parquet()
        with self.assertRaises(json.JSONDecodeError):
            self.run_preprocessing()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(list(self.output.parent.glob(".post_processed_images.*")), [])

    def test_missing_or_duplicate_parquet_key_preserves_old_output(self):
        self.add_image(1, "a.png")
        self.add_image(1, "b.png")
        self.write_parquet()
        self.run_preprocessing()
        before = self.snapshot()
        saved_rows = self.rows[:]
        self.rows = saved_rows[:1]
        self.write_parquet()
        with self.assertRaisesRegex(ValueError, "Missing Parquet metadata"):
            self.run_preprocessing()
        self.assertEqual(before, self.snapshot())
        self.rows = saved_rows + saved_rows[:1]
        self.write_parquet()
        with self.assertRaisesRegex(ValueError, "Duplicate Parquet image key"):
            self.run_preprocessing()
        self.assertEqual(before, self.snapshot())

    def test_copy_failure_preserves_old_output_and_cleans_staging(self):
        self.add_image(1, "a.png")
        self.write_parquet()
        self.run_preprocessing()
        before = self.snapshot()
        with patch.object(preprocessing.shutil, "copy2", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(OSError, "disk failure"):
                self.run_preprocessing()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(list(self.output.parent.glob(".post_processed_images.*")), [])

    def test_publication_failure_restores_old_output(self):
        self.add_image(1, "a.png")
        self.write_parquet()
        self.run_preprocessing()
        before = self.snapshot()
        original_rename = Path.rename

        def fail_publication(path, target):
            if path.name.startswith(".post_processed_images.build-"):
                raise OSError("publication failure")
            return original_rename(path, target)

        with patch.object(Path, "rename", fail_publication):
            with self.assertRaisesRegex(OSError, "publication failure"):
                self.run_preprocessing()
        self.assertEqual(before, self.snapshot())
        self.assertEqual(list(self.output.parent.glob(".post_processed_images.*")), [])


if __name__ == "__main__":
    unittest.main()
