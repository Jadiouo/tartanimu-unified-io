#!/usr/bin/env python3
"""ContextIMUNet -- UnifiedIMUNet's trunk with a cross-window temporal mixer.

The single-window model sees one second of IMU and nothing else, which is enough
for a car or a walking human but not for a hovering drone: in the lowest
angular-rate quintile of drone windows the relative velocity error is 0.963, so
the network is barely beating a constant zero prediction.  A hover simply does
not write the platform's velocity into one second of accelerometer and gyro --
the specific force is dominated by gravity and the residual signal is noise.
The neighbouring seconds do carry it, because a drone that is genuinely
stationary looks different over ten seconds from one drifting slowly, even
though the two are indistinguishable inside any single window.  The official
baseline consumes seq_len=10 consecutive windows for this reason.

The trunk is kept identical to UnifiedIMUNet (same submodule names and shapes),
so a single-window checkpoint loads into it with strict=False and only the mixer
starts from scratch.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unified.model import ResidualBlock


class ContextIMUNet(nn.Module):
    def __init__(self, mixer: str = "gru", max_len: int = 64, hidden: int = 128,
                 width: float = 1.0, dropout: float = 0.2, trunk: str = "cnn"):
        super().__init__()
        self.trunk_kind = trunk
        # width scales the trunk's channels; 1.0 is the published TCMPIO trunk,
        # which every result through round 5 used
        c1, c2, c3, c4 = (max(8, int(round(c * width))) for c in (64, 96, 128, 192))
        if trunk == "cnn":
            self.encoder = nn.Sequential(
                nn.Conv1d(6, c1, 7, padding=3, bias=False), nn.BatchNorm1d(c1), nn.GELU(),
                ResidualBlock(c1, c1),
                ResidualBlock(c1, c2, stride=2),
                ResidualBlock(c2, c2, dilation=2),
                ResidualBlock(c2, c3, stride=2, dilation=2),
                ResidualBlock(c3, c3, dilation=4),
                ResidualBlock(c3, c4, stride=2, dilation=4),
                ResidualBlock(c4, c4, dilation=4),
            )
        elif trunk == "lstm":
            # a recurrent trunk over the raw 200 frames, for comparison against the
            # convolutional one; c4//2 per direction keeps the pooled width identical
            self.encoder = nn.LSTM(6, c4 // 2, num_layers=2, batch_first=True,
                                   bidirectional=True)
        else:
            raise ValueError(f"trunk must be 'cnn' or 'lstm', got {trunk!r}")
        self.attention = nn.Conv1d(c4, 1, 1)
        self.shared = nn.Sequential(nn.LayerNorm(3 * c4), nn.Linear(3 * c4, 256),
                                    nn.GELU(), nn.Dropout(dropout))
        self.velocity_head = nn.Linear(256, 3)
        self.platform_head = nn.Linear(256, 4)

        self.mixer_kind = mixer
        self.max_len = max_len
        if mixer == "gru":
            self.mixer = nn.GRU(256, hidden, num_layers=2, batch_first=True, bidirectional=True)
            self.mixer_proj = nn.Linear(2 * hidden, 256)
        elif mixer == "attn":
            layer = nn.TransformerEncoderLayer(256, nhead=4, dim_feedforward=512, batch_first=True)
            self.mixer = nn.TransformerEncoder(layer, num_layers=2)
            self.position = nn.Parameter(torch.zeros(max_len, 256))
            nn.init.trunc_normal_(self.position, std=0.02)
        else:
            raise ValueError(f"mixer must be 'gru' or 'attn', got {mixer!r}")
        self.mixer_norm = nn.LayerNorm(256)

    def trunk(self, x):
        """(N, 6, 200) -> (N, 256); with trunk="cnn" this is UnifiedIMUNet's computation."""
        if self.trunk_kind == "lstm":
            x = self.encoder(x.transpose(1, 2))[0].transpose(1, 2)   # (N, c4, 200)
        else:
            x = self.encoder(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        features = torch.cat([(x * weights).sum(-1), x.mean(-1), x.std(-1, unbiased=False)], dim=1)
        return self.shared(features)

    def features_and_heads(self, x):
        """(B, K, 6, 200) -> (feature (B, K, 256), velocity (B, K, 3), logits (B, K, 4)).

        The feature returned is the post-mixer one, so an auxiliary head sees the
        same cross-window context the velocity head does.
        """
        B, K = x.shape[:2]
        features = self.trunk(x.reshape(B * K, *x.shape[2:])).view(B, K, 256)
        if self.mixer_kind == "gru":
            mixed = self.mixer_proj(self.mixer(features)[0])
        else:
            if K > self.max_len:
                raise ValueError(f"segment of {K} windows exceeds max_len={self.max_len}")
            mixed = self.mixer(features + self.position[:K])
        features = self.mixer_norm(features + mixed)
        return features, self.velocity_head(features), self.platform_head(features)

    def forward(self, x):
        _, velocity, platform_logits = self.features_and_heads(x)
        return velocity, platform_logits
