from dataclasses import dataclass
from math import sqrt
from pathlib import Path
import random
from uuid import uuid4

import numpy as np
import torch
import yaml
import wandb
from tqdm.auto import tqdm

from diffusion.checkpoint import save_checkpoint
from diffusion.dataloader import create_vae_dataloader
from diffusion.vae import VAE
from diffusion.wandb_logger import log_to_wandb
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from diffusion.vae import VAEOutput


DEFAULT_CONFIG_PATH = "diffusion/config/vae_train_config.yaml"
LOSS_KEYS = ("loss", "recon_mse", "kl", "weighted_kl")


@dataclass
class TrainingProgress:
    global_step: int = 0
    samples_seen: int = 0


def get_config(config_path):
    if not config_path:
        config_path = DEFAULT_CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_dataloaders(config_path=None):
    config = get_config(config_path)
    dataset_config = config["train"]["dataset"]
    train_dataloader = create_vae_dataloader(
        **{
            key: value
            for key, value in dataset_config.items()
            if key not in {"train_dataset", "batch_method"}
        },
    )

    val_dataset_config = config["val"]["dataset"]
    val_dataloader = create_vae_dataloader(
        **{
            key: value
            for key, value in val_dataset_config.items()
            if key not in {"val_dataset", "batch_method"}
        },
    )
    return train_dataloader, val_dataloader


class VAELoss(nn.Module):
    def __init__(self, kl_weight=1e-4):
        super().__init__()
        self.kl_weight = float(kl_weight)

    def forward(self, outputs: VAEOutput, targets):
        reconstruction = outputs.reconstruction.float()
        targets = targets.float()

        # Average pixel error separately for each image.
        recon_loss = (
            (reconstruction - targets)
            .square()
            .flatten(1)
            .mean(dim=1)
        )

        mean = outputs.posterior.mean.float()
        logvar = outputs.posterior.logvar.float()

        # Gaussian KL to N(0, I), normalized per latent element.
        kl_loss = (
            0.5 * (mean.square() + logvar.exp() - 1.0 - logvar)
        ).flatten(1).mean(dim=1)

        weighted_kl = self.kl_weight * kl_loss
        # Each component is [B]; the objective's normalization is unchanged.
        return {
            "loss": recon_loss + weighted_kl,
            "recon_mse": recon_loss,
            "kl": kl_loss,
            "weighted_kl": weighted_kl,
        }


def train_one_epoch(model: VAE,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: VAELoss,
                    device: torch.device,
                    accumulation_samples=64,
                    *,
                    progress=None,
                    run=None,
                    epoch=1,
                    log_every_steps=50):

    if accumulation_samples <= 0:
        raise ValueError("accumulation_samples must be greater than 0")
    if log_every_steps <= 0:
        raise ValueError("log_every_steps must be greater than 0")
    if progress is None:
        progress = TrainingProgress()

    model.train()
    optimizer.zero_grad(set_to_none=True)

    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    interval_totals = dict.fromkeys(LOSS_KEYS, 0.0)
    interval_samples = 0
    total_samples = 0
    pending_samples = 0

    def log_interval():
        nonlocal interval_samples
        if interval_samples == 0:
            return
        metrics = {
            f"train/{key}": value / interval_samples
            for key, value in interval_totals.items()
        }
        metrics.update({
            "train/lr": optimizer.param_groups[0]["lr"],
            "train/samples_seen": progress.samples_seen,
        })
        log_to_wandb(
            run, metrics=metrics, global_step=progress.global_step, epoch=epoch,
        )
        interval_totals.update(dict.fromkeys(LOSS_KEYS, 0.0))
        interval_samples = 0

    def step(sample_count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        progress.global_step += 1
        if progress.global_step % log_every_steps == 0:
            log_interval()

    training_bar = tqdm(
        dataloader, desc=f"Train epoch {epoch}", unit="batch", dynamic_ncols=True,
    )
    for batch in training_bar:
        batch = batch.to(device, non_blocking=True)
        offset = 0

        while offset < batch.shape[0]:
            # Split a batch when necessary to fill exactly one accumulation window

            count = min(accumulation_samples - pending_samples, batch.shape[0] - offset)

            images = batch[offset:offset + count]

            losses = criterion(
                model(images, sample_posterior=True), images
            )
            losses["loss"].sum().backward()

            sums = torch.stack([
                losses[key].detach().sum() for key in LOSS_KEYS
            ]).cpu().tolist()
            for key, value in zip(LOSS_KEYS, sums):
                totals[key] += value
                interval_totals[key] += value
            total_samples += count
            interval_samples += count
            progress.samples_seen += count
            pending_samples += count
            offset += count

            if pending_samples == accumulation_samples:
                step(pending_samples)
                pending_samples = 0

        training_bar.set_postfix(
            loss=f"{totals['loss'] / total_samples:.4f}",
            images=total_samples,
            refresh=False,
        )

    # Retain the final incomplete accumulation window
    if pending_samples > 0:
        step(pending_samples)

    if total_samples == 0:
        raise ValueError("Training loader is empty")

    # Flush the remaining interval even for epochs shorter than the log cadence.
    log_interval()
    return {key: value / total_samples for key, value in totals.items()}


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    totals = dict.fromkeys(LOSS_KEYS, 0.0)
    total_samples = 0
    latent_count = 0
    mean_sum = mean_square_sum = posterior_std_sum = variance_sum = 0.0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch, sample_posterior=False)
            losses = criterion(outputs, batch)

            sums = torch.stack([losses[key].sum() for key in LOSS_KEYS]).cpu().tolist()
            for key, value in zip(LOSS_KEYS, sums):
                totals[key] += value
            total_samples += batch.shape[0]

            mean = outputs.posterior.mean.double()
            variance = outputs.posterior.logvar.double().exp()
            moments = torch.stack([
                mean.sum(), mean.square().sum(), variance.sqrt().sum(), variance.sum(),
            ]).cpu().tolist()
            mean_sum += moments[0]
            mean_square_sum += moments[1]
            posterior_std_sum += moments[2]
            variance_sum += moments[3]
            latent_count += mean.numel()

    if total_samples == 0:
        raise ValueError("Validation loader is empty")

    metrics = {key: value / total_samples for key, value in totals.items()}
    mean_average = mean_sum / latent_count
    mean_variance = max(0.0, mean_square_sum / latent_count - mean_average ** 2)
    # Exact aggregate posterior moments, without consuming training RNG draws.
    # Var(z) = Var(mu) + E[sigma^2] for z sampled from the posterior.
    metrics.update({
        "posterior_mean": mean_average,
        "posterior_mean_std": sqrt(mean_variance),
        "posterior_std": posterior_std_sum / latent_count,
        "sampled_latent_std": sqrt(mean_variance + variance_sum / latent_count),
    })
    return metrics


def get_preview_examples(dataloader, count):
    """Select fixed examples across resolution groups, respecting any subset."""
    if count < 0:
        raise ValueError("num_images must be nonnegative")
    groups = list(dataloader.batch_sampler.resolution_groups.values())
    if count == 0 or not groups:
        return []
    if len(groups) > count:
        positions = np.linspace(0, len(groups) - 1, count, dtype=int)
        groups = [groups[position] for position in positions]
    indices = []
    offset = 0
    while len(indices) < count:
        added = False
        for group in groups:
            if offset < len(group):
                indices.append(group[offset])
                added = True
                if len(indices) == count:
                    break
        if not added:
            break
        offset += 1
    return [(index, dataloader.dataset[index]) for index in indices]


@torch.no_grad()
def reconstruct_examples(model, examples, device):
    model.eval()
    pairs = []
    for index, original in examples:
        reconstruction = model(
            original.unsqueeze(0).to(device), sample_posterior=False,
        ).reconstruction[0].detach().cpu()
        height, width = original.shape[-2:]
        pairs.append((f"validation index {index}, {width}x{height}", original, reconstruction))
    return pairs


def train(config_path=None):
    config = get_config(config_path)
    logging_config = config.get("wandb", {})
    checkpoint_config = config.setdefault("checkpoint", {})
    selection_metric = checkpoint_config.setdefault("selection_metric", "val/recon_mse")
    if selection_metric not in {f"val/{key}" for key in LOSS_KEYS}:
        raise ValueError("checkpoint.selection_metric must name a validation loss component")
    log_every_steps = logging_config.get("log_every_steps", 50)
    if log_every_steps <= 0:
        raise ValueError("wandb.log_every_steps must be greater than 0")
    num_images = logging_config.get("num_images", 8)
    if num_images < 0:
        raise ValueError("wandb.num_images must be nonnegative")
    save_every_epochs = checkpoint_config.get("save_every_epochs", 0)
    if save_every_epochs < 0:
        raise ValueError("checkpoint.save_every_epochs must be nonnegative")
    accumulation_samples = config["train"].get("accumulation_samples", 64)
    if accumulation_samples <= 0:
        raise ValueError("train.accumulation_samples must be greater than 0")

    seed = config["train"].get("seed", 0)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_dataloader, val_dataloader = get_dataloaders(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"input_dim": 3, "latent_dim": 4, "base_channels": 64}
    model_kwargs.update(config.get("model_kwargs", {}))
    model = VAE(**model_kwargs).to(device)
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = VAELoss(kl_weight=config["train"]["kl_weight"])

    examples = get_preview_examples(val_dataloader, num_images)
    config["resolved"] = {
        "model_kwargs": model_kwargs,
        "optimizer": "Adam",
        "optimizer_defaults": optimizer.defaults,
        "seed": seed,
        "accumulation_samples": accumulation_samples,
        "train_samples": sum(map(len, train_dataloader.batch_sampler.resolution_groups.values())),
        "val_samples": sum(map(len, val_dataloader.batch_sampler.resolution_groups.values())),
        "preview_indices": [index for index, _ in examples],
        "train_sample_posterior": True,
        "val_sample_posterior": False,
        "loss_reduction": "mean pixels/latent elements per image, then mean images",
        "latent_statistics_reduction": "all validation latent elements",
    }

    run_id = uuid4().hex[:12]
    checkpoint_dir = Path(checkpoint_config.get("dir", "checkpoints/vae")) / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    wandb_dir = Path(logging_config.get("dir", "runs/wandb"))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best.pt"
    best_metric = float("inf")
    best_epoch = None
    progress = TrainingProgress()

    with wandb.init(
        project=logging_config.get("project", "vlafactory-vae"),
        entity=logging_config.get("entity"),
        name=logging_config.get("name"),
        mode=logging_config.get("mode", "online"),
        dir=str(wandb_dir),
        id=run_id,
        config=config,
    ) as run:
        run.define_metric("global_step")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="global_step")
        for prefix in ("train_epoch", "val", "latent", "recon", "checkpoint"):
            run.define_metric(f"{prefix}/*", step_metric="epoch")
        run.define_metric(selection_metric, step_metric="epoch", summary="min")

        log_to_wandb(
            run, metrics={}, global_step=0, epoch=0,
            reconstruction_pairs=(
                reconstruct_examples(model, examples, device) if not run.disabled else None
            ),
        )
        print(f"Checkpoints: {checkpoint_dir}")
        num_epochs = config["train"]["epochs"]
        for epoch in range(num_epochs):
            train_dataloader.batch_sampler.set_epoch(epoch)
            train_metrics = train_one_epoch(
                model, train_dataloader, optimizer, criterion, device,
                accumulation_samples=accumulation_samples,
                progress=progress, run=run, epoch=epoch + 1,
                log_every_steps=log_every_steps,
            )
            val_metrics = validate_one_epoch(model, val_dataloader, criterion, device)
            metric_value = val_metrics[selection_metric.removeprefix("val/")]
            is_best = metric_value < best_metric
            if is_best:
                best_metric = metric_value
                best_epoch = epoch + 1

            checkpoint_state = {
                "model": model,
                "optimizer": optimizer,
                "epoch": epoch + 1,
                "global_step": progress.global_step,
                "samples_seen": progress.samples_seen,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "config": config,
                "wandb_run_id": run.id if not run.disabled else None,
            }
            last_path = save_checkpoint(checkpoint_dir / "last.pt", **checkpoint_state)
            if is_best:
                save_checkpoint(best_path, **checkpoint_state)
            if save_every_epochs and (epoch + 1) % save_every_epochs == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch + 1:04d}.pt", **checkpoint_state)

            metrics = {f"train_epoch/{key}": value for key, value in train_metrics.items()}
            metrics.update({
                f"val/{key}": val_metrics[key] for key in LOSS_KEYS
            })
            metrics.update({
                f"latent/{key}": value for key, value in val_metrics.items() if key not in LOSS_KEYS
            })
            metrics.update({
                "checkpoint/best_score": best_metric,
                "checkpoint/best_epoch": best_epoch,
                "checkpoint/last_path": str(last_path),
            })
            if best_path.exists():
                metrics["checkpoint/best_path"] = str(best_path)
            log_to_wandb(
                run, metrics=metrics, global_step=progress.global_step, epoch=epoch + 1,
                reconstruction_pairs=(
                    reconstruct_examples(model, examples, device) if not run.disabled else None
                ),
            )
            run.summary.update({
                "best_validation_metric": selection_metric,
                "best_validation_score": best_metric,
                "best_epoch": best_epoch,
                "checkpoint_dir": str(checkpoint_dir),
            })
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val MSE: {val_metrics['recon_mse']:.4f}"
            )

        if checkpoint_config.get("upload_best", False) and best_path.exists() and not run.disabled:
            artifact = wandb.Artifact(
                f"vae-{run_id}", type="model",
                metadata={"metric": selection_metric, "score": best_metric, "epoch": best_epoch},
            )
            artifact.add_file(str(best_path), name="best.pt")
            run.log_artifact(artifact, aliases=["best", "latest"])

if __name__ == "__main__":
    train()
