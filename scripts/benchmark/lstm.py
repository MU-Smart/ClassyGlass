"""
Vanilla LSTM classifier for Dataset_1A — model definition only.
Run via:  python run.py --model lstm --window 500 --step 250

A 2-layer bidirectional LSTM with dropout, taking the raw
(time × channels) sequence and classifying from the final hidden state.
"""

import torch
import torch.nn as nn

# Tells the runner to keep each window as a 2-D sequence (time × channels)
FLATTEN_INPUT = False


class LSTM(nn.Module):
    """
    2-layer bidirectional LSTM → linear classifier.
    Input shape: (batch, window_size, n_channels)
    """

    def __init__(
        self,
        n_channels:  int   = 6,
        n_classes:   int   = 11,
        hidden_size: int   = 128,
        num_layers:  int   = 2,
        dropout:     float = 0.5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.drop = nn.Dropout(dropout)
        # bidirectional → 2 × hidden_size
        self.fc   = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, n_channels)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers * 2, batch, hidden_size)
        # concat last forward and last backward hidden states
        fwd = h_n[-2]          # (batch, hidden_size)
        bwd = h_n[-1]          # (batch, hidden_size)
        x   = torch.cat([fwd, bwd], dim=1)   # (batch, hidden_size * 2)
        x   = self.drop(x)
        return self.fc(x)


def build_model(n_channels: int, n_classes: int) -> nn.Module:
    return LSTM(n_channels=n_channels, n_classes=n_classes)
