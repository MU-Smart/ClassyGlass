"""
CNN classifier for Dataset_1A — model definition only.
Run via:  python run.py --model cnn --window 500 --step 250
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Tells the runner to flatten each window to a 1-D feature vector
FLATTEN_INPUT = True


class CNN(nn.Module):
    """1-D CNN on the flat window vector (window_size × n_channels)."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.fc1   = nn.Linear(128, 128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3   = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size * n_channels)  — flat
        x = x.unsqueeze(1)                    # (batch, 1, n_features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)          # (batch, 128)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)


def build_model(_n_channels: int, n_classes: int) -> nn.Module:
    return CNN(n_classes)
