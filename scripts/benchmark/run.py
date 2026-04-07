"""
Unified runner — data loading, LOSO cross-validation, training, and visualisation.
Model files contain only the nn.Module definition.

Usage:
  python run.py --model cnn --window 500 --step 250
  python run.py --model deep_conv_lstm --window 500 --step 250 --gpu 1

Arguments:
  --model   {cnn, deep_conv_lstm}   Model to run (required)
  --window  int                     Sliding window size in samples (default: 500)
  --step    int                     Sliding window step in samples  (default: 250)
  --gpu     int                     CUDA device index (default: 0).
                                    Ignored when CUDA unavailable — falls back to
                                    MPS then CPU.
"""

import argparse
import glob
import importlib
import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# ─── Registry ─────────────────────────────────────────────────────────────────
# To add a new model: create its file with FLATTEN_INPUT and build_model(),
# then add one entry here.

MODELS: dict[str, str] = {
    "cnn":            "cnn",
    "deep_conv_lstm": "deep_conv_lstm",
}

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(__file__)
DATASET_PATH = os.path.join(_HERE, "..", "..", "Processed-DataSets", "Dataset_1A")
LOGS_DIR     = os.path.join(_HERE, "logs")
MODELS_DIR   = os.path.join(_HERE, "models")
FIGURES_DIR  = os.path.join(_HERE, "figures")

for _d in (LOGS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("runner")
logger.setLevel(logging.DEBUG)
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)
logger.addHandler(_console_handler)

# ─── Constants ────────────────────────────────────────────────────────────────

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
EPOCHS      = 100
RANDOM_SEED = 42
VAL_SPLIT   = 0.1
LR          = 1e-3
LR_FACTOR   = 0.5
LR_PATIENCE = 4
ES_PATIENCE = 8


# ─── 1. Data loading ──────────────────────────────────────────────────────────

def _exp_no(exp_id: int) -> int:
    return exp_id % 11 or 11


def load_windows(window_size: int, step_size: int,
                 dataset_path: str = DATASET_PATH):
    """
    Returns
    -------
    X           : (n_windows, window_size, N_CHANNELS)  float32  — always 3-D
    y           : (n_windows,)  int  activity labels 1–11
    subject_ids : (n_windows,)  int
    """
    all_files = glob.glob(os.path.join(dataset_path, "*", "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found under: {dataset_path}")

    rows = []
    for path in all_files:
        fname  = os.path.basename(path)
        parts  = fname.split("_")
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
        ], axis=1)  # (n_samples, 6)

        label = _exp_no(exp_id)
        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(raw[start : start + window_size])   # (W, 6)
            labels.append(label)
            subjects.append(subject_id)

    X           = np.array(windows,  dtype=np.float32)   # (n, W, 6)
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
    """
    Wraps (n, W, C) windows.
    flatten=True  → yields (W*C,)  vectors  (for CNN)
    flatten=False → yields (W, C)  sequences (for DeepConvLSTM etc.)
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, flatten: bool):
        data = X.reshape(len(X), -1) if flatten else X
        self.X = torch.from_numpy(data).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── 3. Training utilities ────────────────────────────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device):
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
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model(X_batch)
        total_loss += criterion(logits, y_batch).item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


def _fit(model, X_train: np.ndarray, y_train: np.ndarray,
         flatten: bool, device: torch.device, model_path: str):
    """Train with validation split, early stopping, and LR scheduling."""
    dataset = WindowDataset(X_train, y_train, flatten)
    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    trn_len = len(dataset) - val_len
    trn_set, val_set = random_split(
        dataset, [trn_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    trn_loader = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_loss = float("inf")
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        trn_loss, trn_acc = _train_one_epoch(model, trn_loader, criterion, optimizer, device)
        val_loss, val_acc = _evaluate(model, val_loader, criterion, device)
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

    model.load_state_dict(torch.load(model_path + ".tmp", map_location=device))


# ─── 4. LOSO cross-validation ────────────────────────────────────────────────

def run_loso(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
             build_model_fn, flatten: bool,
             model_path: str, device: torch.device):
    le    = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y))

    unique_subjects  = np.unique(subject_ids)
    fold_accuracies: list[float] = []
    all_y_true:      list[int]   = []
    all_y_pred:      list[int]   = []
    best_acc = -1.0

    n_tr_full, W, C = X.shape

    for subject in unique_subjects:
        test_mask  = subject_ids == subject
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y_enc[train_mask]
        y_test      = np.asarray(y_enc[test_mask])

        # Scale per-channel across all time steps
        scaler = StandardScaler()
        n_tr   = X_train_raw.shape[0]
        X_train_sc = scaler.fit_transform(
            X_train_raw.reshape(-1, C)
        ).reshape(n_tr, W, C).astype(np.float32)
        X_test_sc = scaler.transform(
            X_test_raw.reshape(-1, C)
        ).reshape(X_test_raw.shape[0], W, C).astype(np.float32)

        logger.info("─" * 60)
        logger.info(
            f"LOSO fold — held-out: User{subject} "
            f"({test_mask.sum()} test windows, {train_mask.sum()} train windows)"
        )

        model = build_model_fn(C, N_CLASSES).to(device)
        _fit(model, X_train_sc, y_train, flatten, device, model_path)

        model.eval()
        with torch.no_grad():
            inp    = torch.from_numpy(
                X_test_sc.reshape(len(X_test_sc), -1) if flatten else X_test_sc
            ).float().to(device)
            y_pred = model(inp).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        logger.info(f"  User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)
            logger.info(f"  ✓ New best model saved (User{subject} held out, acc={acc:.4f})")

    tmp = model_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    logger.info(f"Best model: {model_path}  (acc={best_acc:.4f})")
    return fold_accuracies, np.array(all_y_true), np.array(all_y_pred), le


# ─── 5. Visualisation ────────────────────────────────────────────────────────

def print_report(fold_accuracies, y_true, y_pred, le,
                 subject_ids_unique, model_name: str, tag: str):
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

    title = model_name.replace("_", " ").title()

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 9))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{title} – Confusion Matrix (LOSO)  [{tag}]")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"{model_name}_confusion_matrix_{tag}.png")
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
    plt.title(f"{title} – Per-subject LOSO Accuracy  [{tag}]")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, f"{model_name}_per_subject_accuracy_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    logger.info(f"Per-subject accuracy chart saved to: {bar_path}")
    plt.close()


# ─── 6. Device selection ──────────────────────────────────────────────────────

def get_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if gpu >= n:
            logger.warning(f"Requested GPU {gpu} but only {n} device(s) available — using GPU 0.")
            gpu = 0
        return torch.device(f"cuda:{gpu}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.warning("CUDA not available — using MPS.")
        return torch.device("mps")
    logger.warning("CUDA/MPS not available — using CPU.")
    return torch.device("cpu")


# ─── 7. CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a model against Dataset_1A with LOSO cross-validation."
    )
    parser.add_argument("--model",  required=True, choices=list(MODELS),
                        help=f"Model to run: {{{', '.join(MODELS)}}}")
    parser.add_argument("--window", type=int, default=500,
                        help="Sliding window size in samples (default: 500)")
    parser.add_argument("--step",   type=int, default=250,
                        help="Sliding window step size in samples (default: 250)")
    parser.add_argument("--gpu",    type=int, default=0,
                        help="CUDA device index (default: 0)")
    return parser.parse_args()


def main():
    
    args   = parse_args()
    device = get_device(args.gpu)
    tag    = f"W{args.window}_S{args.step}"

    # Attach log file
    log_path      = os.path.join(LOGS_DIR, f"{args.model}_{tag}.log")
    file_handler  = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(_fmt)
    logger.addHandler(file_handler)

    # Load model module
    sys.path.insert(0, _HERE)
    module     = importlib.import_module(MODELS[args.model])
    flatten    = module.FLATTEN_INPUT
    model_path = os.path.join(MODELS_DIR, f"{args.model}_best_{tag}.pth")

    logger.info("=" * 60)
    logger.info(f"{args.model}  |  window={args.window}  step={args.step}")
    logger.info(f"Device  : {device}")
    logger.info(f"Flatten : {flatten}")
    logger.info(f"Log     : {log_path}")
    logger.info(f"Model   : {model_path}")
    logger.info(f"Figures : {FIGURES_DIR}")
    logger.info("=" * 60)

    torch.manual_seed(RANDOM_SEED)

    X, y, subject_ids = load_windows(args.window, args.step)
    fold_accuracies, y_true, y_pred, le = run_loso(
        X, y, subject_ids,
        build_model_fn=module.build_model,
        flatten=flatten,
        model_path=model_path,
        device=device,
    )
    print_report(fold_accuracies, y_true, y_pred, le,
                 np.unique(subject_ids), args.model, tag)

    logger.info("=" * 60)
    logger.info(f"{args.model} training complete")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
