"""Log precomputed VAE metrics and reconstruction comparisons."""

import torch
import wandb


def log_to_wandb(
    run,
    *,
    metrics,
    global_step,
    epoch,
    reconstruction_pairs=None,
):
    """Log once; pairs contain a caption and two detached CHW image tensors.

    Inputs and reconstructions use the dataset's [-1, 1] convention. Only
    display copies are clamped; the original tensors and metrics are untouched.
    """
    if run is None or run.disabled:
        return

    payload = dict(metrics, global_step=global_step, epoch=epoch)
    if reconstruction_pairs:
        for index, (caption, original, reconstruction) in enumerate(reconstruction_pairs):
            pair = torch.cat(
                (original.detach().float().cpu(), reconstruction.detach().float().cpu()),
                dim=-1,
            )
            pixels = (
                pair.add(1).div(2).clamp(0, 1).mul(255).round()
                .to(torch.uint8).permute(1, 2, 0).numpy()
            )
            # Separate keys preserve native sizes: W&B image lists expect all
            # images to share one resolution.
            payload[f"recon/example_{index:02d}"] = wandb.Image(
                pixels, caption=f"{caption} | original (left), reconstruction (right)",
            )

    # W&B's internal history step advances automatically. These explicit
    # counters are custom chart axes, so epoch logs cannot collide with steps.
    run.log(payload)
