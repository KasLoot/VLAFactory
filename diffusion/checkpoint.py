"""Save complete VAE training state independently of W&B."""

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch,
    global_step,
    samples_seen,
    best_metric,
    config,
    best_epoch=None,
    wandb_run_id=None,
):
    """Atomically save state after a completed epoch, returning its path.

    ``epoch`` is the number of completed epochs; resume at that zero-based
    sampler epoch. Checkpoints include optimizer and Python/NumPy RNG objects,
    so a future loader must use weights_only=False for these trusted files.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_dim": model.input_dim,
            "latent_dim": model.latent_dim,
            "base_channels": model.encoder[0].out_channels,
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "samples_seen": samples_seen,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "config": config,
        "wandb_run_id": wandb_run_id,
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    # The temporary file must share the destination filesystem for os.replace.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as file:
            temporary_path = Path(file.name)
            torch.save(checkpoint, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path
