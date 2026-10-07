"""Exercise VAE logging, accumulation, and checkpoint persistence on small inputs."""

import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch
from torch import nn
import yaml

from diffusion.checkpoint import save_checkpoint
from diffusion.train_vae import (
    TrainingProgress, VAELoss, train,
    train_one_epoch, validate_one_epoch,
)
from diffusion.vae import GaussianPosterior, VAE, VAEOutput
from diffusion.wandb_logger import log_to_wandb


class TinyVAE(nn.Module):
    """A scalar model whose accumulation can be compared with direct updates."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, images, sample_posterior=True):
        mean = images[:, :1] * self.weight
        posterior = GaussianPosterior(mean, torch.zeros_like(mean))
        return VAEOutput(images * self.weight, posterior, mean)


class TrainingTests(unittest.TestCase):
    def test_loss_components_keep_per_image_normalization(self):
        targets = torch.zeros(2, 3, 8, 16)
        reconstruction = torch.stack([
            torch.ones(3, 8, 16), torch.full((3, 8, 16), 2.0),
        ])
        mean = torch.stack([torch.ones(1, 1, 2), torch.full((1, 1, 2), 2.0)])
        output = VAEOutput(reconstruction, GaussianPosterior(mean, torch.zeros_like(mean)), mean)
        components = VAELoss(kl_weight=0.2)(output, targets)
        torch.testing.assert_close(components["recon_mse"], torch.tensor([1.0, 4.0]))
        torch.testing.assert_close(components["kl"], torch.tensor([0.5, 2.0]))
        torch.testing.assert_close(components["weighted_kl"], torch.tensor([0.1, 0.4]))
        torch.testing.assert_close(components["loss"], torch.tensor([1.1, 4.4]))

    def test_accumulation_matches_direct_updates_and_flushes_partial_window(self):
        images = torch.arange(1, 8, dtype=torch.float32).reshape(7, 1, 1, 1).expand(-1, 3, 2, 2)
        model = TinyVAE()
        reference = TinyVAE()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.001)
        criterion = VAELoss(kl_weight=0.1)
        expected_sums = {key: 0.0 for key in ("loss", "recon_mse", "kl", "weighted_kl")}
        for group in images.split(3):
            components = criterion(reference(group), group)
            for key in expected_sums:
                expected_sums[key] += components[key].detach().sum().item()
            reference_optimizer.zero_grad()
            components["loss"].mean().backward()
            reference_optimizer.step()

        progress = TrainingProgress()
        with patch("diffusion.train_vae.log_to_wandb") as logger:
            metrics = train_one_epoch(
                model, [images[:2], images[2:6], images[6:]], optimizer,
                criterion, torch.device("cpu"), accumulation_samples=3,
                progress=progress, epoch=1, log_every_steps=2,
            )
        torch.testing.assert_close(model.weight, reference.weight)
        for key in metrics:
            self.assertAlmostEqual(metrics[key], expected_sums[key] / 7, places=5)
        self.assertEqual((progress.global_step, progress.samples_seen), (3, 7))
        self.assertEqual([call.kwargs["global_step"] for call in logger.call_args_list], [2, 3])
        self.assertEqual(
            [call.kwargs["metrics"]["train/samples_seen"] for call in logger.call_args_list],
            [6, 7],
        )
        for key, value in metrics.items():
            logs = [call.kwargs["metrics"][f"train/{key}"] for call in logger.call_args_list]
            self.assertAlmostEqual((6 * logs[0] + logs[1]) / 7, value, places=5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_validation_weights_images_and_reports_posterior_moments(self):
        model = TinyVAE()
        small = torch.ones(2, 3, 2, 2)
        large = torch.full((1, 3, 2, 4), 2.0)
        before = torch.get_rng_state().clone()
        metrics = validate_one_epoch(model, [small, large], VAELoss(0), torch.device("cpu"))
        self.assertAlmostEqual(metrics["recon_mse"], (2 * 0.75 ** 2 + 1.5 ** 2) / 3)
        # Equal latent-element counts from the two batches: mu=0.25 and mu=0.5.
        self.assertAlmostEqual(metrics["posterior_mean"], 0.375)
        self.assertAlmostEqual(metrics["posterior_mean_std"], 0.125)
        self.assertAlmostEqual(metrics["posterior_std"], 1.0)
        self.assertAlmostEqual(metrics["sampled_latent_std"], (1 + 0.125 ** 2) ** 0.5)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        self.assertFalse(model.training)

    def test_progress_continues_across_epochs_and_empty_loaders_fail(self):
        model = TinyVAE()
        optimizer = torch.optim.SGD(model.parameters(), lr=0)
        criterion = VAELoss()
        progress = TrainingProgress()
        for epoch in (1, 2):
            train_one_epoch(
                model, [torch.ones(2, 3, 2, 2)], optimizer, criterion,
                torch.device("cpu"), accumulation_samples=3,
                progress=progress, epoch=epoch,
            )
        self.assertEqual((progress.global_step, progress.samples_seen), (2, 4))
        with self.assertRaisesRegex(ValueError, "empty"):
            train_one_epoch(model, [], optimizer, criterion, torch.device("cpu"))
        with self.assertRaisesRegex(ValueError, "empty"):
            validate_one_epoch(model, [], criterion, torch.device("cpu"))


class LoggerTests(unittest.TestCase):
    def test_logs_different_resolutions_without_modifying_sources(self):
        run = Mock(disabled=False)
        originals = [torch.full((3, 2, 3), -1.0), torch.full((3, 4, 2), 1.0)]
        reconstructions = [torch.full_like(originals[0], 3.0), torch.full_like(originals[1], -3.0)]
        with patch("diffusion.wandb_logger.wandb.Image", side_effect=lambda pixels, **kw: pixels) as image:
            log_to_wandb(
                run, metrics={"val/loss": 1.2}, global_step=5, epoch=2,
                reconstruction_pairs=[
                    (str(index), original, reconstruction)
                    for index, (original, reconstruction) in enumerate(zip(originals, reconstructions))
                ],
            )
        payload = run.log.call_args.args[0]
        self.assertEqual((payload["global_step"], payload["epoch"]), (5, 2))
        self.assertEqual(payload["val/loss"], 1.2)
        self.assertEqual(payload["recon/example_00"].shape, (2, 6, 3))
        self.assertEqual(payload["recon/example_01"].shape, (4, 4, 3))
        self.assertTrue(np.all(image.call_args_list[0].args[0][:, :3] == 0))
        self.assertTrue(np.all(image.call_args_list[0].args[0][:, 3:] == 255))
        self.assertTrue(torch.all(reconstructions[0] == 3))
        self.assertTrue(torch.all(reconstructions[1] == -3))
        self.assertEqual(run.log.call_count, 1)

    def test_disabled_logging_skips_media_conversion(self):
        run = Mock(disabled=True)
        with patch("diffusion.wandb_logger.wandb.Image") as image:
            log_to_wandb(run, metrics={}, global_step=0, epoch=0, reconstruction_pairs=[None])
            log_to_wandb(None, metrics={}, global_step=0, epoch=0)
        run.log.assert_not_called()
        image.assert_not_called()


class CheckpointTests(unittest.TestCase):
    def test_roundtrip_contains_optimizer_config_progress_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            model = VAE(base_channels=2)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            model(torch.ones(1, 3, 8, 8)).reconstruction.square().mean().backward()
            optimizer.step()
            cpu_rng = torch.get_rng_state().clone()
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            path = save_checkpoint(
                Path(directory) / "run" / "last.pt", model=model, optimizer=optimizer,
                epoch=2, global_step=4, samples_seen=7, best_metric=0.3,
                config={"checkpoint": {"selection_metric": "val/recon_mse"}},
                best_epoch=1, wandb_run_id="abc",
            )
            state = torch.load(path, map_location="cpu", weights_only=False)
            restored = VAE(**state["model_config"])
            restored.load_state_dict(state["model_state_dict"])
            restored_optimizer = torch.optim.Adam(restored.parameters())
            restored_optimizer.load_state_dict(state["optimizer_state_dict"])
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, restored.state_dict()[key])
            self.assertTrue(restored_optimizer.state)
            self.assertEqual((state["epoch"], state["global_step"], state["samples_seen"]), (2, 4, 7))
            self.assertEqual(state["best_metric"], 0.3)
            self.assertEqual(state["best_epoch"], 1)
            self.assertEqual(state["wandb_run_id"], "abc")
            self.assertEqual(state["rng_state"]["python"], python_rng)
            np.testing.assert_array_equal(state["rng_state"]["numpy"][1], numpy_rng[1])
            torch.testing.assert_close(state["rng_state"]["torch"], cpu_rng)
            self.assertEqual(list(path.parent.iterdir()), [path])

    def test_interrupted_write_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "last.pt"
            path.write_bytes(b"previous checkpoint")
            model = VAE(base_channels=2)
            with patch("diffusion.checkpoint.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    save_checkpoint(
                        path, model=model, optimizer=torch.optim.Adam(model.parameters()),
                        epoch=1, global_step=1, samples_seen=2, best_metric=0.1, config={},
                    )
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(path.parent.iterdir()), [path])


class IntegrationTests(unittest.TestCase):
    def test_offline_training_saves_best_last_history_and_image_media(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            records = []
            for index, size in enumerate([(8, 8), (8, 8), (16, 8)]):
                name = f"{index}.png"
                Image.new("RGB", size, (index * 60,) * 3).save(dataset / name)
                records.append({"image": name, "width": size[0], "height": size[1]})
            (dataset / "label.json").write_text(json.dumps(records))
            dataset_config = {
                "dataset_path": str(dataset), "num_workers": 0,
                "pin_memory": False, "max_batch_size": 2,
            }
            config = {
                "model": "VAE", "model_kwargs": {"base_channels": 2},
                "train": {
                    "epochs": 2, "lr": 0.001, "kl_weight": 0.0001,
                    "accumulation_samples": 2, "dataset": dict(dataset_config, shuffle=True),
                },
                "val": {"dataset": dict(dataset_config, shuffle=False, indices=[1, 2])},
                "wandb": {
                    "mode": "offline", "dir": str(root / "wandb"),
                    "num_images": 2, "log_every_steps": 1,
                },
                "checkpoint": {"dir": str(root / "checkpoints"), "save_every_epochs": 1},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            with patch("torch.cuda.is_available", return_value=False):
                train(config_path)
            run_dirs = list((root / "checkpoints").iterdir())
            self.assertEqual(len(run_dirs), 1)
            run_dir = run_dirs[0]
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {"last.pt", "best.pt", "epoch_0001.pt", "epoch_0002.pt"},
            )
            last = torch.load(run_dir / "last.pt", weights_only=False)
            best = torch.load(run_dir / "best.pt", weights_only=False)
            self.assertEqual((last["epoch"], last["global_step"], last["samples_seen"]), (2, 4, 6))
            self.assertEqual(last["config"]["resolved"]["val_samples"], 2)
            self.assertEqual(set(last["config"]["resolved"]["preview_indices"]), {1, 2})
            self.assertEqual(best["best_metric"], last["best_metric"])
            self.assertTrue(list((root / "wandb").rglob("*.wandb")))
            # Baseline plus both completed epochs, with two separate image pairs.
            self.assertEqual(len(list((root / "wandb").rglob("*.png"))), 6)

    def test_disabled_training_keeps_best_when_validation_worsens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            Image.new("RGB", (8, 8)).save(dataset / "image.png")
            (dataset / "label.json").write_text(json.dumps([
                {"image": "image.png", "width": 8, "height": 8},
            ]))
            dataset_config = {"dataset_path": str(dataset), "num_workers": 0, "pin_memory": False}
            config = {
                "model_kwargs": {"base_channels": 2},
                "train": {"epochs": 2, "lr": 0.001, "kl_weight": 0.0001, "dataset": dataset_config},
                "val": {"dataset": dict(dataset_config, shuffle=False)},
                "wandb": {"mode": "disabled", "dir": str(root / "wandb"), "num_images": 0},
                "checkpoint": {"dir": str(root / "checkpoints"), "selection_metric": "val/loss"},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config))
            scores = iter([0.1, 0.2])

            def validation_with_known_scores(*args):
                metrics = validate_one_epoch(*args)
                metrics["loss"] = next(scores)
                return metrics

            with (
                patch("torch.cuda.is_available", return_value=False),
                patch("diffusion.train_vae.validate_one_epoch", side_effect=validation_with_known_scores),
            ):
                train(config_path)
            run_dir = next((root / "checkpoints").iterdir())
            best = torch.load(run_dir / "best.pt", weights_only=False)
            last = torch.load(run_dir / "last.pt", weights_only=False)
            self.assertEqual((best["epoch"], last["epoch"]), (1, 2))
            self.assertEqual(last["best_metric"], 0.1)
            self.assertEqual(last["best_epoch"], 1)
            self.assertIsNone(last["wandb_run_id"])
            self.assertEqual({path.name for path in run_dir.iterdir()}, {"last.pt", "best.pt"})


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
