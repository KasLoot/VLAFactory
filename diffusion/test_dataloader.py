"""Check resolution batching, sample coverage, and normalized image loading."""

from collections import Counter
import json
from pathlib import Path
import random
import shutil
import tempfile
import unittest

from PIL import Image
import torch

from diffusion.dataloader import ResolutionBatchSampler, create_vae_dataloader
from diffusion.dataset import VAEDataset


class ResolutionBatchSamplerTests(unittest.TestCase):
    def setUp(self):
        self.records = (
            [{"width": 8, "height": 8} for _ in range(7)]
            + [{"width": 8, "height": 16} for _ in range(3)]
            + [{"width": 32, "height": 16}]
        )

    def test_every_image_once_with_matching_dimensions_and_partial_batches(self):
        sampler = ResolutionBatchSampler(
            self.records, max_batch_pixels=256, max_batch_size=3,
        )
        batches = list(sampler)
        self.assertEqual(len(sampler), len(batches))
        self.assertEqual(len(batches), 6)
        self.assertEqual(sorted(index for batch in batches for index in batch),
                         list(range(len(self.records))))
        counts = Counter()
        for batch in batches:
            resolutions = {(self.records[i]["width"], self.records[i]["height"])
                           for i in batch}
            self.assertEqual(len(resolutions), 1)
            width, height = resolutions.pop()
            counts[(width, height)] += len(batch)
            self.assertLessEqual(len(batch), 3)
            if width * height <= 256:
                self.assertLessEqual(len(batch) * width * height, 256)
            else:
                self.assertEqual(len(batch), 1)
        self.assertEqual(counts, {(8, 8): 7, (8, 16): 3, (32, 16): 1})

    def test_epoch_shuffle_is_reproducible_and_preserves_global_rng(self):
        records = self.records * 10
        first = ResolutionBatchSampler(records, max_batch_size=3, seed=42)
        second = ResolutionBatchSampler(records, max_batch_size=3, seed=42)
        python_state = random.getstate()
        torch_state = torch.get_rng_state().clone()
        epoch_zero = list(first)
        self.assertEqual(epoch_zero, list(first))
        self.assertEqual(epoch_zero, list(second))
        first.set_epoch(1)
        second.set_epoch(1)
        self.assertEqual(list(first), list(second))
        self.assertNotEqual(epoch_zero, list(first))
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_subset_preserves_original_dataset_indices(self):
        sampler = ResolutionBatchSampler(self.records, indices=[10, 1, 7, 0])
        self.assertEqual(sorted(i for batch in sampler for i in batch), [0, 1, 7, 10])

    def test_validation_order_does_not_change_with_epoch(self):
        sampler = ResolutionBatchSampler(self.records, max_batch_size=3, shuffle=False)
        expected = [[0, 1, 2], [3, 4, 5], [6], [7, 8, 9], [10]]
        self.assertEqual(list(sampler), expected)
        sampler.set_epoch(100)
        self.assertEqual(list(sampler), expected)

    def test_empty_subset(self):
        sampler = ResolutionBatchSampler(self.records, indices=[])
        self.assertEqual(len(sampler), 0)
        self.assertEqual(list(sampler), [])

    def test_invalid_indices_and_resolutions_are_rejected(self):
        for indices in ([0, 0], [-1], [11], [True], [0.5]):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                ResolutionBatchSampler(self.records, indices=indices)
        for record in ({}, {"width": 0, "height": 8},
                       {"width": True, "height": 8},
                       {"width": 8, "height": "16"}):
            with self.subTest(record=record), self.assertRaisesRegex(ValueError, "index 0"):
                ResolutionBatchSampler([record])

    def test_invalid_budgets_and_epochs_are_rejected(self):
        for options in ({"max_batch_pixels": 0}, {"max_batch_size": -1},
                        {"max_batch_size": True}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ResolutionBatchSampler(self.records, **options)
        sampler = ResolutionBatchSampler(self.records)
        with self.assertRaises(ValueError):
            sampler.set_epoch(-1)


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="vae-dataloader-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        records = []
        for index, (size, color) in enumerate([
            ((8, 8), 0), ((8, 8), 255), ((8, 8), 127),
            ((16, 8), 64), ((16, 8), 128), ((8, 16), 192),
        ]):
            name = f"{index}.png"
            with Image.new("RGB", size, (color, color, color)) as image:
                image.save(self.root / name)
            records.append({"image": name, "width": size[0], "height": size[1]})
        (self.root / "post_processed_images-labels.json").write_text(json.dumps(records))

    def test_factory_loads_normalized_images_without_workers(self):
        loader = create_vae_dataloader(
            dataset_path=self.root, max_batch_size=2, num_workers=0,
            pin_memory=False, shuffle=False,
        )
        batches = list(loader)
        self.assertEqual(len(loader), 4)
        self.assertEqual(sum(len(batch) for batch in batches), 6)
        for batch in batches:
            self.assertEqual(batch.ndim, 4)
            self.assertEqual(batch.shape[1], 3)
            self.assertEqual(batch.dtype, torch.float32)
            self.assertGreaterEqual(batch.min().item(), -1)
            self.assertLessEqual(batch.max().item(), 1)
        self.assertTrue(torch.equal(batches[0][0], torch.full((3, 8, 8), -1.0)))
        self.assertTrue(torch.equal(batches[0][1], torch.full((3, 8, 8), 1.0)))

    def test_subset_and_multiworker_loading(self):
        loader = create_vae_dataloader(
            self.root, indices=[0, 3, 5], num_workers=2,
            pin_memory=False, persistent_workers=False,
        )
        self.assertEqual(loader.dataset.dataset_path, self.root)
        batches = list(loader)
        self.assertEqual(sum(len(batch) for batch in batches), 3)
        self.assertEqual({tuple(batch.shape[-2:]) for batch in batches},
                         {(8, 8), (8, 16), (16, 8)})

    def test_factory_rejects_invalid_worker_settings(self):
        for options in ({"num_workers": -1}, {"prefetch_factor": 0}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_vae_dataloader(dataset_path=self.root, **options)

    def test_separate_train_val_directories_and_both_manifest_names(self):
        original = json.loads((self.root / "post_processed_images-labels.json").read_text())
        for split, indices, manifest_name in (
            ("train", [0, 1, 3], "label.json"),
            ("val", [2, 4], "post_processed_images-labels.json"),
        ):
            directory = self.root / split
            directory.mkdir()
            records = []
            for local_index, original_index in enumerate(indices):
                record = dict(original[original_index])
                name = f"{local_index}.png"
                shutil.copy2(self.root / record["image"], directory / name)
                record["image"] = name
                records.append(record)
            (directory / manifest_name).write_text(json.dumps(records))

        train = create_vae_dataloader(
            self.root / "train", shuffle=True, num_workers=0, pin_memory=False,
        )
        val = create_vae_dataloader(
            self.root / "val", shuffle=False, num_workers=0, pin_memory=False,
        )
        self.assertEqual(train.dataset.dataset_path, self.root / "train")
        self.assertEqual(val.dataset.dataset_path, self.root / "val")
        self.assertEqual(len(train.dataset), 3)
        self.assertEqual(len(val.dataset), 2)
        self.assertTrue(train.batch_sampler.shuffle)
        self.assertFalse(val.batch_sampler.shuffle)
        self.assertEqual(sum(len(batch) for batch in train), 3)
        self.assertEqual(sum(len(batch) for batch in val), 2)
        train_black = train.dataset[0]
        val_gray = val.dataset[0]
        self.assertTrue(torch.all(train_black == -1))
        self.assertFalse(torch.equal(train_black, val_gray))

    def test_shuffle_is_explicit_for_given_directory(self):
        loader = create_vae_dataloader(
            self.root, shuffle=False,
            num_workers=0, pin_memory=False,
        )
        self.assertFalse(loader.batch_sampler.shuffle)
        first = list(loader.batch_sampler)
        loader.batch_sampler.set_epoch(1)
        self.assertEqual(list(loader.batch_sampler), first)

    def test_split_path_is_required_for_dataset_and_loader(self):
        for factory in (VAEDataset, create_vae_dataloader):
            with self.subTest(factory=factory), self.assertRaisesRegex(ValueError, "dataset_path"):
                factory()
            for path in (None, "", "   "):
                with self.subTest(factory=factory, path=path):
                    with self.assertRaisesRegex(ValueError, "dataset_path"):
                        factory(path)

    def test_parent_directory_is_not_expanded_into_a_split(self):
        parent = self.root / "parent"
        train = parent / "train"
        train.mkdir(parents=True)
        records = json.loads((self.root / "post_processed_images-labels.json").read_text())[:1]
        shutil.copy2(self.root / "0.png", train / "0.png")
        (train / "label.json").write_text(json.dumps(records))
        with self.assertRaisesRegex(FileNotFoundError, "label.json"):
            create_vae_dataloader(parent, num_workers=0)
        loader = create_vae_dataloader(train, num_workers=0, pin_memory=False)
        self.assertEqual(len(loader.dataset), 1)
        self.assertEqual(next(iter(loader)).shape, (1, 3, 8, 8))

    def test_dataset_converts_grayscale_and_rgba_to_rgb(self):
        with Image.new("L", (8, 8), 128) as image:
            image.save(self.root / "0.png")
        with Image.new("RGBA", (8, 8), (255, 0, 0, 255)) as image:
            image.save(self.root / "1.png")
        dataset = VAEDataset(self.root)
        gray, rgba = dataset[0], dataset[1]
        self.assertEqual(gray.shape, (3, 8, 8))
        self.assertEqual(rgba.shape, (3, 8, 8))
        self.assertTrue(torch.equal(gray[0], gray[1]))
        self.assertTrue(torch.all(rgba[0] == 1))
        self.assertTrue(torch.all(rgba[1:] == -1))


if __name__ == "__main__":
    unittest.main()
