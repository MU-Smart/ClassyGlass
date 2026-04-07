"""
CNN classifier for Dataset_1A — PyTorch implementation.

Pipeline:
  Raw sensor CSVs (Datasets/Dataset_1A)
  → sliding windows (window_size samples, step_size stride) over accel + gyro axes
  → flatten each window to a 1-D feature vector (window_size × 6 raw values)
  → Leave-One-Subject-Out (LOSO) cross-validation
      train on all subjects except one, test on the held-out subject
      repeat for every subject → report mean accuracy across subjects
  → 1-D CNN → 11-class softmax

Device priority: CUDA → MPS → CPU

Usage:
  python cnn.py --window 500 --step 250
"""

import argparse
import glob
import logging
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(__file__)
DATASET_PATH = os.path.join(_HERE, "..", "..", "Datasets", "Dataset_1A")
LOGS_DIR     = os.path.join(_HERE, "logs")
MODELS_DIR   = os.path.join(_HERE, "models")
FIGURES_DIR  = os.path.join(_HERE, "figures")

for _d in (LOGS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Logger (file handler added in main after args are parsed) ────────────────

logger = logging.getLogger("cnn")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)
logger.addHandler(_console_handler)

# ─── Fixed constants ──────────────────────────────────────────────────────────

ACTIVITY_NAMES = {
    1:  "Sitting – Reading",
    2:  "Sitting – Writing",
    3:  "Computer – Typing",
    4:  "Computer – Browsing",
    5:  "Sitting – Moving head/body",
    6:  "Sitting – Moving chair",
    7:  "Stand up from sitting",
    8:  "Standing",
    9:  "Walking",
    10: "Running",
    11: "Taking stairs",
}

N_CHANNELS  = 6
N_CLASSES   = 11
BATCH_SIZE  = 32
EPOCHS      = 50
RANDOM_SEED = 42
VAL_SPLIT   = 0.1       # fraction of training data used for validation
LR          = 1e-3
LR_FACTOR   = 0.5
LR_PATIENCE = 4
ES_PATIENCE = 8          # early-stopping patience (epochs)


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── 1. Raw data loading ──────────────────────────────────────────────────────

def _exp_no(exp_id: int) -> int:
    return exp_id % 11 or 11


def load_raw_windows(window_size: int, step_size: int,
                     dataset_path: str = DATASET_PATH):
    """
    Returns
    -------
    X           : (n_windows, window_size * N_CHANNELS)  float32
    y           : (n_windows,)  int  activity labels 1–11
    subject_ids : (n_windows,)  int  user number from folder name
    """
    all_files = glob.glob(os.path.join(dataset_path, "*", "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found under: {dataset_path}")

    rows = []
    for path in all_files:
        fname = os.path.basename(path)
        parts = fname.split("_")
        if len(parts) < 6:
            continue
        try:
            exp_id = int(parts[0])
        except ValueError:
            continue
        sensor      = parts[4]
        user_folder = os.path.basename(os.path.dirname(path))
        try:
            subject_id = int(user_folder.replace("User", ""))
        except ValueError:
            continue
        rows.append({"path": path, "expID": exp_id, "sensor": sensor,
                     "subject_id": subject_id})

    meta    = pd.DataFrame(rows)
    windows, labels, subjects = [], [], []
    skipped = 0

    for exp_id, group in meta.groupby("expID"):
        if not {"Accelerometer", "Gyroscope"}.issubset(set(group["sensor"].values)):
            skipped += 1
            continue

        acc_path   = group.loc[group["sensor"] == "Accelerometer", "path"].iloc[0]
        gyr_path   = group.loc[group["sensor"] == "Gyroscope",     "path"].iloc[0]
        subject_id = group["subject_id"].iloc[0]

        acc = pd.read_csv(acc_path, usecols=["x-axis (g)",     "y-axis (g)",     "z-axis (g)"])
        gyr = pd.read_csv(gyr_path, usecols=["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"])

        n_samples = min(len(acc), len(gyr))
        raw = np.concatenate([
            acc.values[:n_samples].astype(np.float32),
            gyr.values[:n_samples].astype(np.float32),
        ], axis=1)

        label = _exp_no(exp_id)
        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(raw[start : start + window_size].flatten())
            labels.append(label)
            subjects.append(subject_id)

    X           = np.array(windows,  dtype=np.float32)
    y           = np.array(labels,   dtype=int)
    subject_ids = np.array(subjects, dtype=int)

    logger.info(
        f"Loaded {X.shape[0]} windows | window={window_size}  step={step_size} | "
        f"{meta['expID'].nunique()} experiments | "
        f"{len(np.unique(subject_ids))} subjects | "
        f"{skipped} experiments skipped"
    )
    return X, y, subject_ids


# ─── 2. Dataset ───────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── 3. CNN model ─────────────────────────────────────────────────────────────

class CNN(nn.Module):
    """1-D CNN on the flat window vector."""

    def __init__(self, n_classes: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1,  64,  kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)   # → (batch, 128, 1)
        self.fc1   = nn.Linear(128, 128)
        self.drop1 = nn.Dropout(0.4)
        self.fc2   = nn.Linear(128, 64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3   = nn.Linear(64,  n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)                          # (batch, 1, n_features)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)                # (batch, 128)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = F.relu(self.fc2(x))
        x = self.drop2(x)
        return self.fc3(x)                          # logits


# ─── 4. Training utilities ────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


def fit(model, X_train: np.ndarray, y_train: np.ndarray,
        device: torch.device, model_path: str):
    """Train with validation split, early stopping, and LR scheduling."""
    dataset  = WindowDataset(X_train, y_train)
    val_len  = max(1, int(len(dataset) * VAL_SPLIT))
    trn_len  = len(dataset) - val_len
    trn_set, val_set = random_split(
        dataset, [trn_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    trn_loader = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        trn_loss, trn_acc = train_one_epoch(model, trn_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            f"  epoch {epoch:3d}  "
            f"loss {trn_loss:.4f}  acc {trn_acc:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path + ".tmp")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                logger.info(f"  Early stopping at epoch {epoch}")
                break

    # Restore best weights for this fold
    model.load_state_dict(torch.load(model_path + ".tmp", map_location=device))


# ─── 5. LOSO cross-validation ────────────────────────────────────────────────

def run_loso(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
             model_path: str, device: torch.device):
    le    = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y))

    unique_subjects  = np.unique(subject_ids)
    fold_accuracies: list[float] = []
    all_y_true:      list[int]   = []
    all_y_pred:      list[int]   = []
    best_acc = -1.0

    for subject in unique_subjects:
        test_mask  = subject_ids == subject
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y_enc[train_mask]
        y_test      = np.asarray(y_enc[test_mask])

        scaler     = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train_raw)
        X_test_sc  = scaler.transform(X_test_raw)

        logger.info("─" * 60)
        logger.info(
            f"LOSO fold — held-out: User{subject} "
            f"({test_mask.sum()} test windows, {train_mask.sum()} train windows)"
        )

        model = CNN(N_CLASSES).to(device)
        fit(model, X_train_sc, y_train, device, model_path)

        # Predict on held-out subject
        model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X_test_sc).float().to(device)
            y_pred   = model(X_tensor).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        logger.info(f"  User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            logger.info(
                f"  ✓ New best model saved (User{subject} held out, acc={acc:.4f})"
            )

    # Clean up temp file
    tmp = model_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    logger.info(f"Best model: {model_path}  (acc={best_acc:.4f})")
    return fold_accuracies, np.array(all_y_true), np.array(all_y_pred), le


# ─── 6. Reporting ─────────────────────────────────────────────────────────────

def print_report(fold_accuracies, y_true, y_pred, le,
                 subject_ids_unique, figures_dir: str, tag: str):
    class_names = [ACTIVITY_NAMES[le.classes_[i]] for i in range(len(le.classes_))]

    logger.info("=" * 60)
    logger.info("LOSO SUMMARY")
    logger.info("=" * 60)
    for subject, acc in zip(subject_ids_unique, fold_accuracies):
        logger.info(f"  User{subject}: {acc:.4f}")
    logger.info(f"  Mean accuracy : {np.mean(fold_accuracies):.4f}")
    logger.info(f"  Std  accuracy : {np.std(fold_accuracies):.4f}")

    logger.info("=" * 60)
    logger.info("CLASSIFICATION REPORT (all folds combined)")
    logger.info("=" * 60)
    logger.info("\n" + str(classification_report(y_true, y_pred, target_names=class_names)))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"CNN – Confusion Matrix (LOSO)  [{tag}]")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(figures_dir, f"confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150)
    logger.info(f"Confusion matrix saved to: {cm_path}")
    plt.close()

    # Per-subject accuracy bar chart
    plt.figure(figsize=(10, 5))
    plt.bar([f"User{s}" for s in subject_ids_unique], fold_accuracies,
            color="steelblue", edgecolor="white")
    plt.axhline(np.mean(fold_accuracies), color="tomato", linewidth=1.5,
                linestyle="--", label=f"Mean {np.mean(fold_accuracies):.3f}")
    plt.ylabel("Accuracy")
    plt.title(f"CNN – Per-subject LOSO Accuracy  [{tag}]")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(figures_dir, f"per_subject_accuracy_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    logger.info(f"Per-subject accuracy chart saved to: {bar_path}")
    plt.close()


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="CNN + LOSO on Dataset_1A (PyTorch)")
    parser.add_argument("--window", type=int, default=500,
                        help="Sliding window size in samples (default: 500)")
    parser.add_argument("--step",   type=int, default=250,
                        help="Sliding window step size in samples (default: 250)")
    return parser.parse_args()


def main():
    args        = parse_args()
    window_size = args.window
    step_size   = args.step
    tag         = f"W{window_size}_S{step_size}"
    log_path    = os.path.join(LOGS_DIR,   f"cnn_{tag}.log")
    model_path  = os.path.join(MODELS_DIR, f"best_model_{tag}.pth")

    _file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    _file_handler.setFormatter(_fmt)
    logger.addHandler(_file_handler)

    torch.manual_seed(RANDOM_SEED)
    device = get_device()

    logger.info("=" * 60)
    logger.info(f"CNN + LOSO (PyTorch)  |  window={window_size}  step={step_size}")
    logger.info(f"Device : {device}")
    logger.info(f"Log    : {log_path}")
    logger.info(f"Model  : {model_path}")
    logger.info(f"Figures: {FIGURES_DIR}")
    logger.info("=" * 60)

    X, y, subject_ids = load_raw_windows(window_size, step_size)
    fold_accuracies, y_true, y_pred, le = run_loso(
        X, y, subject_ids, model_path, device
    )
    print_report(fold_accuracies, y_true, y_pred, le,
                 np.unique(subject_ids), FIGURES_DIR, tag)

    logger.info("=" * 60)
    logger.info("CNN + LOSO training complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
