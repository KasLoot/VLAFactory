import torch
import torch.functional as F

import os
import json
import yaml
import wandb

from diffusion.dataloader import create_vae_dataloader
from diffusion.vae import VAE
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from diffusion.vae import VAEOutput


DEFAULT_CONFIG_PATH = "diffusion/config/vae_train_config.yaml"

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

        return recon_loss + self.kl_weight * kl_loss  # [B]


def train_one_epoch(model: VAE,
                    dataloader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    criterion: VAELoss,
                    device: torch.device,
                    accumulation_samples=64):

    if accumulation_samples <= 0:
        raise ValueError("accumulation_samples must be greater than 0")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0
    pending_samples = 0

    def step(sample_count):
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(sample_count)

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    for batch in dataloader:
        batch = batch.to(device, non_blocking=True)
        offset = 0

        while offset < batch.shape[0]:
            # Split a batch when necessary to fill exactly one accumulation window

            count = min(accumulation_samples - pending_samples, batch.shape[0] - offset)

            images = batch[offset:offset + count]

            losses = criterion(
                model(images, sample_posterior=True), images
                )
            losses.sum().backward()

            total_loss += losses.detach().sum().item()
            total_samples += count
            pending_samples += count
            offset += count

            if pending_samples == accumulation_samples:
                step(pending_samples)
                pending_samples = 0

    # Retain the final incomplete accumulation window
    if pending_samples > 0:
        step(pending_samples)

    if total_samples == 0:
        raise ValueError("Training loader is empty")

    return total_loss / total_samples

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch, sample_posterior=False)
            losses = criterion(outputs, batch)

            total_loss += losses.sum().item()
            total_samples += batch.shape[0]

    if total_samples == 0:
        raise ValueError("Validation loader is empty")

    return total_loss / total_samples


def train():
    config = get_config("diffusion/config/vae_train_config.yaml")

    train_dataloader, val_dataloader = get_dataloaders("diffusion/config/vae_train_config.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE().to(device)
    optimizer = Adam(model.parameters(), lr=config["train"]["lr"])
    criterion = VAELoss(kl_weight=config["train"]["kl_weight"])

    num_epochs = config["train"]["epochs"]
    for epoch in range(num_epochs):
        train_dataloader.batch_sampler.set_epoch(epoch)
        train_loss = train_one_epoch(model, train_dataloader, optimizer, criterion, device)
        val_loss = validate_one_epoch(model, val_dataloader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        # wandb.log({"epoch": epoch+1, "train_loss": train_loss, "val_loss": val_loss})

if __name__ == "__main__":
    train()
