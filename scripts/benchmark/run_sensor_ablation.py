"""
Sensor-modality ablation for deep_conv_lstm.

Answers: which sensors contribute to performance, and is the magnetometer
useful as a direct feature input (vs. only for drift correction)?

Modalities tested (each sensor in isolation)
--------------------------------------------
  acc   Accelerometer   3 ch  @ 100 Hz
  gyro  Gyroscope       3 ch  @ 100 Hz
  mag   Magnetometer    3 ch  @ 20 Hz  → interpolated to 100 Hz
  baro  Pressure        1 ch  @ ~7 Hz  → interpolated to 100 Hz

Sweep
-----
  Windows  : 300, 500
  Overlaps : 0 %, 50 %
  Model    : deep_conv_lstm

Usage:
  python run_sensor_ablation.py
  python run_sensor_ablation.py --gpu 1
  python run_sensor_ablation.py --modalities imu imu_mag   # subset
"""

import argparse
import glob
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset, random_split

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "..", "..", "Processed-DataSets", "Dataset_1A")
LOGS_DIR     = os.path.join(_HERE, "logs")
MODELS_DIR   = os.path.join(_HERE, "models")
FIGURES_DIR  = os.path.join(_HERE, "figures")

for _d in (LOGS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Sensor config ────────────────────────────────────────────────────────────

# Columns to read from each sensor CSV
SENSOR_COLS: dict[str, list[str]] = {
    "Accelerometer": ["x-axis (g)",     "y-axis (g)",     "z-axis (g)"],
    "Gyroscope":     ["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"],
    "Magnetometer":  ["x-axis (T)",     "y-axis (T)",     "z-axis (T)"],
    "Pressure":      ["pressure (Pa)"],
}

# Sensors that run at the base rate (100 Hz); others will be resampled to match.
BASE_RATE_SENSORS = {"Accelerometer", "Gyroscope"}

# Modalities: ordered list of sensors to concatenate.
# Order determines column layout; keep it consistent.
MODALITIES: dict[str, list[str]] = {
    "acc":   ["Accelerometer"],
    "gyro":  ["Gyroscope"],
    "mag":   ["Magnetometer"],
    "baro":  ["Pressure"],
}

# ─── Sweep grid ───────────────────────────────────────────────────────────────

WINDOWS      = [300, 500]
OVERLAP_PCTS = [0, 50]

# ─── Hyper-parameters (mirror run.py) ────────────────────────────────────────

N_CLASSES    = 11
BATCH_SIZE   = 256
NUM_WORKERS  = 4
EPOCHS       = 100
RANDOM_SEED  = 42
VAL_SPLIT    = 0.1
LR           = 1e-3
LR_FACTOR    = 0.5
LR_PATIENCE  = 4
ES_PATIENCE  = 8

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

# ─── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("sensor_ablation")
logger.setLevel(logging.DEBUG)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


def _setup_logger(timestamp: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_fmt)
    logger.addHandler(ch)
    log_path = os.path.join(LOGS_DIR, f"sensor_ablation_{timestamp}.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_fmt)
    logger.addHandler(fh)
    logger.info(f"Log: {log_path}")


# ─── Data loading ─────────────────────────────────────────────────────────────

def _exp_no(exp_id: int) -> int:
    return exp_id % 11 or 11


def _resample(data: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly interpolate `data` (shape: n_samples × n_cols) to target_len rows."""
    src_x = np.linspace(0, 1, len(data))
    dst_x = np.linspace(0, 1, target_len)
    return np.column_stack([np.interp(dst_x, src_x, data[:, c]) for c in range(data.shape[1])])


def load_windows_for_modality(
    sensors: list[str],
    window_size: int,
    step_size: int,
    dataset_path: str = DATASET_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load windowed data for a given sensor subset.

    Lower-rate sensors (Magnetometer, Pressure) are linearly interpolated to
    match the length of the highest-rate sensor in the modality.

    Returns
    -------
    X           : (n_windows, window_size, n_channels)  float32
    y           : (n_windows,)  int  activity labels 1–11
    subject_ids : (n_windows,)  int
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
        present = set(group["sensor"].values)
        if not set(sensors).issubset(present):
            skipped += 1
            continue

        subject_id = group["subject_id"].iloc[0]

        # Load each sensor; determine reference length from base-rate sensors
        sensor_arrays: dict[str, np.ndarray] = {}
        ref_len = None

        for s in sensors:
            path = group.loc[group["sensor"] == s, "path"].iloc[0]
            df   = pd.read_csv(path, usecols=SENSOR_COLS[s])
            arr  = df.values.astype(np.float32)
            sensor_arrays[s] = arr
            if s in BASE_RATE_SENSORS:
                if ref_len is None:
                    ref_len = len(arr)
                else:
                    ref_len = min(ref_len, len(arr))  # align dual base-rate sensors

        # Fall back: if no base-rate sensor in this modality, use shortest
        if ref_len is None:
            ref_len = min(len(a) for a in sensor_arrays.values())

        # Resample non-base-rate (or mismatched base-rate) sensors
        aligned = []
        for s in sensors:
            arr = sensor_arrays[s]
            if len(arr) != ref_len:
                arr = _resample(arr, ref_len)
            aligned.append(arr)

        raw = np.concatenate(aligned, axis=1).astype(np.float32)  # (ref_len, n_ch)
        n_samples = ref_len
        label     = _exp_no(exp_id)

        for start in range(0, n_samples - window_size + 1, step_size):
            windows.append(raw[start : start + window_size])
            labels.append(label)
            subjects.append(subject_id)

    X           = np.array(windows,  dtype=np.float32)
    y           = np.array(labels,   dtype=int)
    subject_ids = np.array(subjects, dtype=int)

    n_ch = X.shape[2] if X.ndim == 3 else 0
    logger.info(
        f"  Loaded {X.shape[0]} windows | sensors={sensors} ({n_ch} ch) | "
        f"window={window_size}  step={step_size} | skipped={skipped} experiments"
    )
    return X, y, subject_ids


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()          # (n, W, C)
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Training ─────────────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        logits = model(Xb)
        total_loss += criterion(logits, yb).item() * len(yb)
        correct    += (logits.argmax(1) == yb).sum().item()
        n          += len(yb)
    return total_loss / n, correct / n


def _fit(model, X_train: np.ndarray, y_train: np.ndarray,
         device: torch.device, model_path: str) -> None:
    dataset = WindowDataset(X_train, y_train)
    val_len = max(1, int(len(dataset) * VAL_SPLIT))
    trn_len = len(dataset) - val_len
    trn_set, val_set = torch.utils.data.random_split(
        dataset, [trn_len, val_len],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    pin = device.type == "cuda"
    trn_loader = DataLoader(trn_set, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKERS, pin_memory=pin,
                            persistent_workers=NUM_WORKERS > 0)

    criterion = torch.nn.CrossEntropyLoss()
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
            f"    epoch {epoch:3d}  loss {trn_loss:.4f}  acc {trn_acc:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path + ".tmp")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= ES_PATIENCE:
                logger.info(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path + ".tmp", map_location=device))


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def run_loso(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
             n_channels: int, device: torch.device,
             model_path: str) -> tuple[list[float], np.ndarray, np.ndarray, LabelEncoder]:
    sys.path.insert(0, _HERE)
    import deep_conv_lstm as dcl

    le    = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y))

    unique_subjects = np.unique(subject_ids)
    fold_accuracies: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    best_acc = -1.0
    _, W, C = X.shape

    for subject in unique_subjects:
        test_mask  = subject_ids == subject
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y_enc[train_mask]
        y_test      = np.asarray(y_enc[test_mask])

        scaler = StandardScaler()
        n_tr   = X_train_raw.shape[0]
        X_train_sc = scaler.fit_transform(
            X_train_raw.reshape(-1, C)
        ).reshape(n_tr, W, C).astype(np.float32)
        X_test_sc = scaler.transform(
            X_test_raw.reshape(-1, C)
        ).reshape(X_test_raw.shape[0], W, C).astype(np.float32)

        logger.info(f"  ─── LOSO fold: User{subject} "
                    f"(test={test_mask.sum()}, train={train_mask.sum()})")

        model = dcl.DeepConvLSTM(n_channels=n_channels, n_classes=N_CLASSES).to(device)
        _fit(model, X_train_sc, y_train, device, model_path)

        model.eval()
        with torch.no_grad():
            inp    = torch.from_numpy(X_test_sc).float().to(device)
            y_pred = model(inp).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        logger.info(f"    User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

    tmp = model_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    return fold_accuracies, np.array(all_y_true), np.array(all_y_pred), le


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        gpu = gpu if gpu < n else 0
        return torch.device(f"cuda:{gpu}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Results table ────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("SENSOR MODALITY ABLATION — SUMMARY")
    logger.info(f"{'Modality':<12} {'Sensors':<38} {'ch':>3} {'W':>5} {'Ovlp':>5}  {'Mean Acc':>9}  {'Std':>6}")
    logger.info("-" * 72)
    for r in sorted(results, key=lambda x: -x["mean_acc"]):
        logger.info(
            f"{r['modality']:<12} {str(r['sensors']):<38} {r['n_channels']:>3} "
            f"{r['window']:>5} {r['overlap_pct']:>4}%  {r['mean_acc']:>9.4f}  {r['std_acc']:>6.4f}"
        )
    logger.info("=" * 72)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensor-modality ablation for deep_conv_lstm.")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument(
        "--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES),
        help="Modalities to run (default: all)",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _setup_logger(timestamp)

    device = get_device(args.gpu)
    logger.info(f"Device : {device}")
    logger.info(f"GPU    : {torch.cuda.get_device_name(device) if device.type == 'cuda' else 'N/A'}")

    configs = [
        (mod, w, o)
        for mod in args.modalities
        for w   in WINDOWS
        for o   in OVERLAP_PCTS
    ]
    logger.info(f"Experiments: {len(configs)}  "
                f"({len(args.modalities)} modalities × {len(WINDOWS)} windows × {len(OVERLAP_PCTS)} overlaps)")

    results: list[dict] = []
    t_start = time.monotonic()

    for mod, window, overlap_pct in configs:
        sensors   = MODALITIES[mod]
        step      = max(1, int(window * (1 - overlap_pct / 100)))
        tag       = f"sensor_{mod}_W{window}_O{overlap_pct}"
        n_channels = sum(len(SENSOR_COLS[s]) for s in sensors)

        logger.info("")
        logger.info("=" * 72)
        logger.info(f"Modality: {mod}  |  sensors: {sensors}  |  {n_channels} ch  |  "
                    f"window={window}  overlap={overlap_pct}%  step={step}")
        logger.info("=" * 72)

        X, y, subject_ids = load_windows_for_modality(sensors, window, step)

        if len(X) == 0:
            logger.warning(f"  No windows loaded for {tag} — skipping.")
            continue

        model_path = os.path.join(MODELS_DIR, f"deep_conv_lstm_{tag}.pth")

        fold_accs, y_true, y_pred, le = run_loso(
            X, y, subject_ids, n_channels, device, model_path
        )

        class_names = [ACTIVITY_NAMES[le.classes_[i]] for i in range(len(le.classes_))]
        logger.info("")
        logger.info(f"  Mean acc: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")
        logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names))

        results.append({
            "modality":   mod,
            "sensors":    sensors,
            "n_channels": n_channels,
            "window":     window,
            "overlap_pct": overlap_pct,
            "mean_acc":   float(np.mean(fold_accs)),
            "std_acc":    float(np.std(fold_accs)),
            "fold_accs":  fold_accs,
        })

    total_elapsed = time.monotonic() - t_start
    logger.info(f"\nTotal sweep time: {total_elapsed / 60:.1f} min")
    _print_summary(results)


if __name__ == "__main__":
    main()
