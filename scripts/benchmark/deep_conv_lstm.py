"""
DeepConvLSTM classifier for Dataset_1A — model definition only.
Run via:  python run.py --model deep_conv_lstm --window 500 --step 250

Reference: Ordóñez & Roggen, "Deep Convolutional and LSTM Recurrent Neural
Networks for Multimodal Wearable Activity Recognition" (Sensors, 2016).
"""

import torch.nn as nn

# Tells the runner to keep each window as a 2-D sequence (time × channels)
FLATTEN_INPUT = False


class DeepConvLSTM(nn.Module):
    """
    4 × Conv1d (along the time axis)  →  2 × LSTM  →  linear classifier.
    Input shape: (batch, window_size, n_channels)
    """

    def __init__(
        self,
        n_channels:  int   = 6,
        n_classes:   int   = 11,
        n_filters:   int   = 64,
        kernel_size: int   = 5,
        lstm_hidden: int   = 128,
        lstm_layers: int   = 2,
        dropout:     float = 0.5,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, n_filters, kernel_size, padding=padding), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size, padding=padding), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size, padding=padding), nn.ReLU(),
            nn.Conv1d(n_filters,  n_filters, kernel_size, padding=padding), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=n_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(lstm_hidden, n_classes)

    def forward(self, x):
        # x: (batch, time, channels)
        x = x.permute(0, 2, 1)       # → (batch, channels, time)
        x = self.conv(x)              # → (batch, n_filters, time)
        x = x.permute(0, 2, 1)       # → (batch, time, n_filters)
        _, (h_n, _) = self.lstm(x)
        x = self.drop(h_n[-1])        # last-layer hidden: (batch, lstm_hidden)
        return self.fc(x)


def build_model(n_channels: int, n_classes: int) -> nn.Module:
    return DeepConvLSTM(n_channels=n_channels, n_classes=n_classes)
