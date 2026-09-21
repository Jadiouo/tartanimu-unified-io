#!/usr/bin/env python3
"""UnifiedIMUNet -- dilated 1-D ResNet with attention pooling.

Architecture ported verbatim from sample_code/ts-tcmpio-unified-temporal-resnet.ipynb
(cell 15).  Single shared trunk, one velocity head; the platform head is an
auxiliary training-time signal only and is not used at inference, so the model
stays a single unified network with no external routing.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_norm(kind: str, channels: int) -> nn.Module:
    """'bn' BatchNorm1d; 'gn' GroupNorm with 8 groups -- statistics per sample,
    so a mixed-platform batch no longer normalises a hovering drone's features
    against a running car's."""
    if kind == "bn":
        return nn.BatchNorm1d(channels)
    if kind == "gn":
        return nn.GroupNorm(min(8, channels), channels)
    raise ValueError(f"norm must be bn or gn, got {kind!r}")


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, norm="bn"):
        super().__init__()
        padding = 2 * dilation
        self.main = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 5, stride=stride, padding=padding,
                      dilation=dilation, bias=False),
            make_norm(norm, out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 5, padding=padding, dilation=dilation, bias=False),
            make_norm(norm, out_channels),
        )
        self.skip = (nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False)
                     if stride != 1 or in_channels != out_channels else nn.Identity())
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(self.main(x) + self.skip(x))


class UnifiedIMUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(6, 64, 7, padding=3, bias=False), nn.BatchNorm1d(64), nn.GELU(),
            ResidualBlock(64, 64),
            ResidualBlock(64, 96, stride=2),
            ResidualBlock(96, 96, dilation=2),
            ResidualBlock(96, 128, stride=2, dilation=2),
            ResidualBlock(128, 128, dilation=4),
            ResidualBlock(128, 192, stride=2, dilation=4),
            ResidualBlock(192, 192, dilation=4),
        )
        self.attention = nn.Conv1d(192, 1, 1)
        self.shared = nn.Sequential(nn.LayerNorm(576), nn.Linear(576, 256), nn.GELU(), nn.Dropout(0.2))
        self.velocity_head = nn.Linear(256, 3)
        self.platform_head = nn.Linear(256, 4)

    def features_and_heads(self, x):
        """(N, 6, 200) -> (feature (N, 256), velocity (N, 3), platform_logits (N, 4)).

        Auxiliary heads need the shared feature, which forward() discards.
        """
        x = self.encoder(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        features = torch.cat([(x * weights).sum(-1), x.mean(-1), x.std(-1, unbiased=False)], dim=1)
        features = self.shared(features)
        return features, self.velocity_head(features), self.platform_head(features)

    def forward(self, x):
        _, velocity, platform_logits = self.features_and_heads(x)
        return velocity, platform_logits


def vector_huber(prediction, target, beta=0.25):
    error = torch.linalg.vector_norm(prediction - target, dim=1)
    return torch.where(error < beta, 0.5 * error.square() / beta, error - 0.5 * beta).mean()
