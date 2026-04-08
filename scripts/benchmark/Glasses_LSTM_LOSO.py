"""LSTM LOSO training script for Dataset 1A.

Converted from Glasses_LSTM_LOSO.ipynb.
All stdout/stderr output is redirected to a run-specific .log file.
"""

import argparse
import glob
import os
import sys
import warnings
from contextlib import redirect_stderr, redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

warnings.filterwarnings("ignore")

# Paths
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start_dir: str) -> str:
    """Find project root by locating Processed-DataSets/Dataset_1A."""
    cur = os.path.abspath(start_dir)
    for _ in range(6):
        probe = os.path.join(cur, "Processed-DataSets", "Dataset_1A")
        if os.path.isdir(probe):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    raise FileNotFoundError(
        "Could not locate project root containing Processed-DataSets/Dataset_1A"
    )


_PROJECT_ROOT = _find_project_root(_HERE)
_BENCHMARK_DIR = os.path.join(_PROJECT_ROOT, "scripts", "benchmark")

DATASET_PATH = os.path.join(_PROJECT_ROOT, "Processed-DataSets", "Dataset_1A")
FIGURES_DIR = os.path.join(_BENCHMARK_DIR, "figures")
MODELS_DIR = os.path.join(_BENCHMARK_DIR, "models")
LOGS_DIR = os.path.join(_BENCHMARK_DIR, "logs")

for _d in (FIGURES_DIR, MODELS_DIR, LOGS_DIR):
    os.makedirs(_d, exist_ok=True)

# Hyper-parameters
N_CHANNELS = 6
N_CLASSES = 11
BATCH_SIZE = 32
RANDOM_SEED = 42
VAL_SPLIT = 0.1
LR = 1e-3
LR_FACTOR = 0.5
LR_PATIENCE = 4
ES_PATIENCE = 8

torch.manual_seed(RANDOM_SEED)

ACTIVITY_NAMES = {
    1: "Sitting - Reading",
    2: "Sitting - Writing",
    3: "Computer - Typing",
    4: "Computer - Browsing",
    5: "Sitting - Moving head/body",
    6: "Sitting - Moving chair",
    7: "Stand up from sitting",
    8: "Standing",
    9: "Walking",
    10: "Running",
    11: "Taking stairs",
}


def _exp_no(exp_id: int) -> int:
    """Map raw experiment number to activity label 1-11."""
    return exp_id % 11 or 11


def load_windows(window_size: int, step_size: int, dataset_path: str = DATASET_PATH):
    """
    Returns
    -------
    X           : (n_windows, window_size, N_CHANNELS) float32
    y           : (n_windows,) int - activity labels 1-11
    subject_ids : (n_windows,) int
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
        sensor = parts[4]
        user_folder = os.path.basename(os.path.dirname(path))
        try:
            subject_id = int(user_folder.replace("User", ""))
        except ValueError:
            continue
        rows.append({"path": path, "expID": exp_id, "sensor": sensor, "subject_id": subject_id})

    meta = pd.DataFrame(rows)
    windows, labels, subjects = [], [], []
    skipped = 0

    for exp_id, group in meta.groupby("expID"):
        if not {"Accelerometer", "Gyroscope"}.issubset(set(group["sensor"].values)):
            skipped += 1
            continue

        acc_path = group.loc[group["sensor"] == "Accelerometer", "path"].iloc[0]
        gyr_path = group.loc[group["sensor"] == "Gyroscope", "path"].iloc[0]
        subject_id = group["subject_id"].iloc[0]

        acc = pd.read_csv(acc_path, usecols=["x-axis (g)", "y-axis (g)", "z-axis (g)"])
        gyr = pd.read_csv(gyr_path, usecols=["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"])

        n_samples = min(len(acc), len(gyr))
        raw = np.concatenate(
            [
                acc.values[:n_samples].astype(np.float32),
                gyr.values[:n_samples].astype(np.float32),
            ],
            axis=1,
        )

        label = _exp_no(exp_id)
        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(raw[start : start + window_size])
            labels.append(label)
            subjects.append(subject_id)

    X = np.array(windows, dtype=np.float32)
    y = np.array(labels, dtype=int)
    subject_ids = np.array(subjects, dtype=int)

    print(f"Loaded {X.shape[0]} windows | window={window_size} step={step_size}")
    print(f"Experiments: {meta['expID'].nunique()} | Subjects: {len(np.unique(subject_ids))}")
    print(f"Skipped (missing sensor): {skipped}")
    return X, y, subject_ids


class WindowDataset(Dataset):
    """Wraps (n, W, C) windows as sequences - no flattening for LSTM."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    """3-layer stacked LSTM classifier."""

    def __init__(self, n_channels: int = 6, n_classes: int = 11, hidden: int = 128, dropout: float = 0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden,
            num_layers=3,
            batch_first=True,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.drop(h_n[-1])
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def build_model(n_channels: int, n_classes: int) -> nn.Module:
    return LSTMClassifier(n_channels=n_channels, n_classes=n_classes)


def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        logits = model(x_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct += (logits.argmax(1) == y_batch).sum().item()
        n += len(y_batch)
    return total_loss / n, correct / n


def fit(model, x_train: np.ndarray, y_train: np.ndarray, device: torch.device, model_path: str, epochs: int):
    """Train with validation split, early stopping, and LR scheduling."""
    dataset = WindowDataset(x_train, y_train)
    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    trn_len = len(dataset) - val_len
    trn_set, val_set = random_split(
        dataset,
        [trn_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    trn_loader = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
    )

    best_val_loss = float("inf")
    no_improve = 0

    for epoch in range(1, epochs + 1):
        trn_loss, trn_acc = _train_one_epoch(model, trn_loader, criterion, optimizer, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  epoch {epoch:3d} "
                f"loss {trn_loss:.4f} acc {trn_acc:.4f} "
                f"val_loss {val_loss:.4f} val_acc {val_acc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path + ".tmp")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path + ".tmp", map_location=device))


def run(window_size: int, step_size: int, epochs: int):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)

    x, y, subject_ids = load_windows(window_size, step_size)
    print("X:", x.shape, " y:", y.shape, " subjects:", np.unique(subject_ids))

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique_subjects = np.unique(subject_ids)
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []
    best_acc = -1.0

    tag = f"W{window_size}_S{step_size}"
    model_path = os.path.join(MODELS_DIR, f"glasses_lstm_loso_best_{tag}.pth")

    _, w, c = x.shape

    for subject in unique_subjects:
        test_mask = subject_ids == subject
        train_mask = ~test_mask

        x_train_raw = x[train_mask]
        x_test_raw = x[test_mask]
        y_train = y_enc[train_mask]
        y_test = y_enc[test_mask]

        scaler = StandardScaler()
        n_tr = x_train_raw.shape[0]
        x_train_sc = scaler.fit_transform(x_train_raw.reshape(-1, c)).reshape(n_tr, w, c).astype(np.float32)
        x_test_sc = scaler.transform(x_test_raw.reshape(-1, c)).reshape(x_test_raw.shape[0], w, c).astype(np.float32)

        print("-" * 60)
        print(
            f"LOSO fold - held-out: User{subject} "
            f"(test={test_mask.sum()} train={train_mask.sum()})"
        )

        model = build_model(c, N_CLASSES).to(device)
        fit(model, x_train_sc, y_train, device, model_path, epochs)

        model.eval()
        with torch.no_grad():
            inp = torch.from_numpy(x_test_sc).float().to(device)
            y_pred = model(inp).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        print(f"  User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            print(f"  New best model saved (acc={acc:.4f}) -> {model_path}")

    tmp_path = model_path + ".tmp"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    print("=" * 60)
    print("LOSO SUMMARY")
    print("=" * 60)
    for subject, acc in zip(unique_subjects, fold_accuracies):
        print(f"  User{subject}: {acc:.4f}")
    print(f"  Mean accuracy : {np.mean(fold_accuracies):.4f}")
    print(f"  Std  accuracy : {np.std(fold_accuracies):.4f}")
    print()

    class_names = [ACTIVITY_NAMES[le.classes_[i]] for i in range(len(le.classes_))]
    print("=" * 60)
    print("CLASSIFICATION REPORT (all folds combined)")
    print("=" * 60)
    print(classification_report(all_y_true, all_y_pred, target_names=class_names))

    cm = confusion_matrix(all_y_true, all_y_pred)
    plt.figure(figsize=(13, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(f"LSTM - Confusion Matrix (LOSO) [{tag}]", fontsize=14)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"lstm_confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150)
    plt.show()
    print(f"Saved: {cm_path}")

    plt.figure(figsize=(10, 5))
    plt.bar(
        [f"User{s}" for s in unique_subjects],
        fold_accuracies,
        color="steelblue",
        edgecolor="white",
    )
    plt.axhline(
        np.mean(fold_accuracies),
        color="tomato",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean {np.mean(fold_accuracies):.3f}",
    )
    plt.ylabel("Accuracy")
    plt.title(f"LSTM - Per-subject LOSO Accuracy [{tag}]")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, f"lstm_per_subject_accuracy_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    plt.show()
    print(f"Saved: {bar_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="LSTM LOSO runner for Dataset_1A")
    parser.add_argument("--window", type=int, default=500, help="Sliding window size")
    parser.add_argument("--step", type=int, default=250, help="Sliding window step")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = f"W{args.window}_S{args.step}"
    log_path = os.path.join(LOGS_DIR, f"glasses_lstm_loso_{tag}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        with redirect_stdout(log_file), redirect_stderr(log_file):
            print("Starting run")
            print(f"Log file: {log_path}")
            print(f"Model directory: {MODELS_DIR}")
            run(window_size=args.window, step_size=args.step, epochs=args.epochs)
            print("Run finished")

    sys.__stdout__.write(f"Run complete. Full output saved to: {log_path}\n")


if __name__ == "__main__":
    main()
