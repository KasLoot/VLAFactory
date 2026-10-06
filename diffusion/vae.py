from dataclasses import dataclass
from math import gcd

import torch
from torch import nn
import torch.nn.functional as F


def normalization(channels):
    # Valid divisors; keep at least two channels per group when possible.
    groups = gcd(channels, min(32, max(1, channels // 2)))
    return nn.GroupNorm(groups, channels, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm1 = normalization(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = normalization(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = (
            nn.Identity() if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(
            F.interpolate(x, scale_factor=2, mode="nearest")
        )


@dataclass
class GaussianPosterior:
    mean: torch.Tensor
    logvar: torch.Tensor

    def sample(self):
        std = torch.exp(0.5 * self.logvar)
        return self.mean + std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean


@dataclass
class VAEOutput:
    reconstruction: torch.Tensor
    posterior: GaussianPosterior
    latent: torch.Tensor


class VAE(nn.Module):
    compression_factor = 8

    def __init__(self, input_dim=3, latent_dim=4, base_channels=64):
        super().__init__()
        if min(input_dim, latent_dim, base_channels) <= 0:
            raise ValueError("Channel counts must be positive")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        channels = [
            base_channels,
            2 * base_channels,
            4 * base_channels,
            4 * base_channels,
        ]

        encoder_layers = [
            nn.Conv2d(input_dim, channels[0], 3, padding=1)
        ]
        current = channels[0]

        for level, width in enumerate(channels):
            encoder_layers.extend([
                ResBlock(current, width),
                ResBlock(width, width),
            ])
            current = width

            if level < len(channels) - 1:
                encoder_layers.append(
                    nn.Conv2d(width, width, 3, stride=2, padding=1)
                )

        # Final channels contain mean and log-variance.
        encoder_layers.extend([
            normalization(current),
            nn.SiLU(),
            nn.Conv2d(current, 2 * latent_dim, 3, padding=1),
        ])
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = [
            nn.Conv2d(latent_dim, channels[-1], 3, padding=1)
        ]
        current = channels[-1]

        for level in reversed(range(len(channels))):
            width = channels[level]
            decoder_layers.extend([
                ResBlock(current, width),
                ResBlock(width, width),
            ])
            current = width

            if level > 0:
                decoder_layers.append(Upsample(width))

        decoder_layers.extend([
            normalization(current),
            nn.SiLU(),
            nn.Conv2d(current, input_dim, 3, padding=1),
        ])
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        if x.ndim != 4 or x.shape[1] != self.input_dim:
            raise ValueError(f"Expected [B, {self.input_dim}, H, W]")

        h, w = x.shape[-2:]
        if h < 8 or w < 8 or h % 8 or w % 8:
            raise ValueError(
                "Height and width must be positive multiples of 8"
            )

        moments = self.encoder(x)
        mean, logvar = moments.chunk(2, dim=1)

        # Keep posterior arithmetic in FP32 under mixed precision.
        return GaussianPosterior(
            mean=mean.float(),
            logvar=logvar.float().clamp(-30.0, 20.0),
        )

    def decode(self, z):
        if z.ndim != 4 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected [B, {self.latent_dim}, H/8, W/8]"
            )
        return self.decoder(z)

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        z = (
            posterior.sample()
            if sample_posterior
            else posterior.mode()
        )

        return VAEOutput(
            reconstruction=self.decode(z),
            posterior=posterior,
            latent=z,
        )