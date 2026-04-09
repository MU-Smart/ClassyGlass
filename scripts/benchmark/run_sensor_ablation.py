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

All experiments run in parallel, one per GPU (round-robin if more configs
than GPUs). Falls back to CPU/MPS if CUDA is unavailable.

Usage:
  python run_sensor_ablation.py
  python run_sensor_ablation.py --workers 2          # override parallelism
  python run_sensor_ablation.py --modalities acc gyro
"""

import argparse
import concurrent.futures
import glob
import logging
import os
import queue
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, "..", "..", "Processed-DataSets", "Dataset_1A")
LOGS_DIR     = os.path.join(_HERE, "logs")
MODELS_DIR   = os.path.join(_HERE, "models")
FIGURES_DIR  = os.path.join(_HERE, "figures")

for _d in (LOGS_DIR, MODELS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

# ─── Sensor config ────────────────────────────────────────────────────────────

SENSOR_COLS: dict[str, list[str]] = {
    "Accelerometer": ["x-axis (g)",     "y-axis (g)",     "z-axis (g)"],
    "Gyroscope":     ["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"],
    "Magnetometer":  ["x-axis (T)",     "y-axis (T)",     "z-axis (T)"],
    "Pressure":      ["pressure (Pa)"],
}

BASE_RATE_SENSORS = {"Accelerometer", "Gyroscope"}

MODALITIES: dict[str, list[str]] = {
    "acc":  ["Accelerometer"],
    "gyro": ["Gyroscope"],
    "mag":  ["Magnetometer"],
    "baro": ["Pressure"],
}

# ─── Sweep grid ───────────────────────────────────────────────────────────────

WINDOWS      = [300, 500]
OVERLAP_PCTS = [0, 50]

# ─── Hyper-parameters ────────────────────────────────────────────────────────

N_CLASSES   = 11
BATCH_SIZE  = 256
NUM_WORKERS = 0   # experiments run in parallel already; per-loader workers cause "too many open files"
EPOCHS      = 100
RANDOM_SEED = 42
VAL_SPLIT   = 0.1
LR          = 1e-3
LR_FACTOR   = 0.5
LR_PATIENCE = 4
ES_PATIENCE = 8

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

# ─── Master logger (stdout + master file) ────────────────────────────────────

_fmt    = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger  = logging.getLogger("sensor_ablation")
logger.setLevel(logging.DEBUG)
logger.propagate = False


def _setup_master_logger(timestamp: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(_fmt)
    logger.addHandler(ch)
    log_path = os.path.join(LOGS_DIR, f"sensor_ablation_{timestamp}.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_fmt)
    logger.addHandler(fh)
    logger.info(f"Master log: {log_path}")


def _make_exp_logger(tag: str) -> tuple[logging.Logger, logging.FileHandler, str]:
    """Create an isolated logger + file handler for one experiment."""
    exp_log_path = os.path.join(LOGS_DIR, f"{tag}.log")
    exp_logger   = logging.getLogger(f"sensor_ablation.{tag}")
    exp_logger.setLevel(logging.DEBUG)
    exp_logger.propagate = False          # don't bubble up to master logger
    fh = logging.FileHandler(exp_log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_fmt)
    exp_logger.addHandler(fh)
    return exp_logger, fh, exp_log_path


# ─── GPU helpers ─────────────────────────────────────────────────────────────

def get_available_gpus() -> list[int]:
    if torch.cuda.is_available():
        n    = torch.cuda.device_count()
        gpus = list(range(n))
        logger.info(f"Found {n} CUDA GPU(s):")
        for i in gpus:
            mem = torch.cuda.get_device_properties(i).total_memory / 2**30
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem:.1f} GiB)")
        return gpus
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.warning("No CUDA GPUs — using MPS (experiments serialised).")
        return [0]
    logger.warning("No CUDA / MPS — using CPU (experiments serialised).")
    return [0]


def _gpu_to_device(gpu: int) -> torch.device:
    if torch.cuda.is_available():
        return torch.device(f"cuda:{gpu}")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Data loading ─────────────────────────────────────────────────────────────

def _exp_no(exp_id: int) -> int:
    return exp_id % 11 or 11


def _resample(data: np.ndarray, target_len: int) -> np.ndarray:
    src_x = np.linspace(0, 1, len(data))
    dst_x = np.linspace(0, 1, target_len)
    return np.column_stack([np.interp(dst_x, src_x, data[:, c])
                            for c in range(data.shape[1])])


def load_windows_for_modality(
    sensors: list[str],
    window_size: int,
    step_size: int,
    exp_logger: logging.Logger,
    dataset_path: str = DATASET_PATH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

        subject_id     = group["subject_id"].iloc[0]
        sensor_arrays: dict[str, np.ndarray] = {}
        ref_len        = None

        for s in sensors:
            path = group.loc[group["sensor"] == s, "path"].iloc[0]
            df   = pd.read_csv(path, usecols=SENSOR_COLS[s])
            arr  = df.values.astype(np.float32)
            sensor_arrays[s] = arr
            if s in BASE_RATE_SENSORS:
                ref_len = min(ref_len, len(arr)) if ref_len else len(arr)

        if ref_len is None:
            ref_len = min(len(a) for a in sensor_arrays.values())

        aligned = []
        for s in sensors:
            arr = sensor_arrays[s]
            if len(arr) != ref_len:
                arr = _resample(arr, ref_len)
            aligned.append(arr)

        raw       = np.concatenate(aligned, axis=1).astype(np.float32)
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
    exp_logger.info(
        f"Loaded {X.shape[0]} windows | sensors={sensors} ({n_ch} ch) | "
        f"window={window_size}  step={step_size} | skipped={skipped} experiments"
    )
    return X, y, subject_ids


# ─── Dataset ──────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
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
         device: torch.device, model_path: str,
         exp_logger: logging.Logger) -> None:
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
        exp_logger.info(
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
                exp_logger.info(f"    Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(model_path + ".tmp", map_location=device))


# ─── LOSO CV ──────────────────────────────────────────────────────────────────

def run_loso(X: np.ndarray, y: np.ndarray, subject_ids: np.ndarray,
             n_channels: int, device: torch.device, model_path: str,
             exp_logger: logging.Logger,
             ) -> tuple[list[float], np.ndarray, np.ndarray, LabelEncoder]:
    sys.path.insert(0, _HERE)
    import deep_conv_lstm as dcl

    le    = LabelEncoder()
    y_enc = np.asarray(le.fit_transform(y))

    unique_subjects = np.unique(subject_ids)
    fold_accuracies: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    best_acc = -1.0
    _, W, C  = X.shape

    for subject in unique_subjects:
        test_mask  = subject_ids == subject
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        X_test_raw  = X[test_mask]
        y_train     = y_enc[train_mask]
        y_test      = np.asarray(y_enc[test_mask])

        scaler     = StandardScaler()
        n_tr       = X_train_raw.shape[0]
        X_train_sc = scaler.fit_transform(
            X_train_raw.reshape(-1, C)
        ).reshape(n_tr, W, C).astype(np.float32)
        X_test_sc  = scaler.transform(
            X_test_raw.reshape(-1, C)
        ).reshape(X_test_raw.shape[0], W, C).astype(np.float32)

        exp_logger.info(f"  ─── LOSO fold: User{subject} "
                        f"(test={test_mask.sum()}, train={train_mask.sum()})")

        model = dcl.DeepConvLSTM(n_channels=n_channels, n_classes=N_CLASSES).to(device)
        _fit(model, X_train_sc, y_train, device, model_path, exp_logger)

        model.eval()
        with torch.no_grad():
            inp    = torch.from_numpy(X_test_sc).float().to(device)
            y_pred = model(inp).argmax(1).cpu().numpy()

        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        exp_logger.info(f"    User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), model_path)

    tmp = model_path + ".tmp"
    if os.path.exists(tmp):
        os.remove(tmp)

    return fold_accuracies, np.array(all_y_true), np.array(all_y_pred), le


# ─── Results table ────────────────────────────────────────────────────────────

def _print_summary(results: list[dict]) -> None:
    logger.info("")
    logger.info("=" * 72)
    logger.info("SENSOR MODALITY ABLATION — SUMMARY")
    logger.info(f"{'Modality':<12} {'ch':>3} {'W':>5} {'Step':>5} {'Ovlp':>5}  {'Mean Acc':>9}  {'Std':>6}")
    logger.info("-" * 72)
    for r in sorted(results, key=lambda x: -x["mean_acc"]):
        logger.info(
            f"{r['modality']:<12} {r['n_channels']:>3} {r['window']:>5} {r['step']:>5} "
            f"{r['overlap_pct']:>4}%  {r['mean_acc']:>9.4f}  {r['std_acc']:>6.4f}"
        )
    logger.info("=" * 72)


# ─── Experiment worker ────────────────────────────────────────────────────────

def _run_experiment(
    mod: str,
    window: int,
    overlap_pct: int,
    gpu: int,
    gpu_q: "queue.Queue[int]",
) -> dict:
    sensors    = MODALITIES[mod]
    step       = max(1, int(window * (1 - overlap_pct / 100)))
    tag        = f"sensor_{mod}_W{window}_S{step}"
    n_channels = sum(len(SENSOR_COLS[s]) for s in sensors)
    device     = _gpu_to_device(gpu)

    exp_logger, exp_fh, exp_log_path = _make_exp_logger(tag)

    logger.info(f"[START]  {tag:<35s}  gpu={gpu}")
    exp_logger.info("=" * 72)
    exp_logger.info(f"Modality: {mod}  |  sensors: {sensors}  |  {n_channels} ch  |  "
                    f"window={window}  overlap={overlap_pct}%  step={step}")
    exp_logger.info(f"Device: {device}  |  Log: {exp_log_path}")
    exp_logger.info("=" * 72)

    t0 = time.monotonic()
    try:
        X, y, subject_ids = load_windows_for_modality(
            sensors, window, step, exp_logger
        )

        if len(X) == 0:
            exp_logger.warning("No windows loaded — skipping.")
            logger.warning(f"[SKIP]   {tag:<35s}  no windows")
            return {}

        model_path = os.path.join(MODELS_DIR, f"deep_conv_lstm_{tag}.pth")

        fold_accs, y_true, y_pred, le = run_loso(
            X, y, subject_ids, n_channels, device, model_path, exp_logger
        )

        class_names = [ACTIVITY_NAMES[le.classes_[i]] for i in range(len(le.classes_))]
        exp_logger.info("")
        exp_logger.info(f"Mean acc: {np.mean(fold_accs):.4f}  Std: {np.std(fold_accs):.4f}")
        exp_logger.info("\n" + classification_report(y_true, y_pred, target_names=class_names))

        elapsed = time.monotonic() - t0
        logger.info(f"[DONE]   {tag:<35s}  "
                    f"acc={np.mean(fold_accs):.4f}  ({elapsed / 60:.1f} min)")

        return {
            "modality":    mod,
            "sensors":     sensors,
            "n_channels":  n_channels,
            "window":      window,
            "overlap_pct": overlap_pct,
            "step":        step,
            "mean_acc":    float(np.mean(fold_accs)),
            "std_acc":     float(np.std(fold_accs)),
            "fold_accs":   fold_accs,
        }

    except Exception as exc:
        elapsed = time.monotonic() - t0
        exp_logger.exception("Experiment failed")
        logger.error(f"[ERROR]  {tag:<35s}  {exc}  ({elapsed / 60:.1f} min)")
        return {}

    finally:
        exp_logger.removeHandler(exp_fh)
        exp_fh.close()
        gpu_q.put(gpu)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sensor-modality ablation for deep_conv_lstm.")
    p.add_argument(
        "--workers", type=int, default=0,
        help="Override number of parallel workers (default: number of GPUs)",
    )
    p.add_argument(
        "--modalities", nargs="+", choices=list(MODALITIES), default=list(MODALITIES),
        help="Modalities to run (default: all)",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _setup_master_logger(timestamp)

    gpus      = get_available_gpus()
    n_workers = args.workers if args.workers > 0 else len(gpus)

    # Round-robin GPU queue
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    configs = [
        (mod, w, o)
        for mod in args.modalities
        for w   in WINDOWS
        for o   in OVERLAP_PCTS
    ]
    n_total = len(configs)

    logger.info("=" * 60)
    logger.info(f"Sensor ablation: {n_total} experiments")
    logger.info(f"  Modalities : {args.modalities}")
    logger.info(f"  Windows    : {WINDOWS}")
    logger.info(f"  Overlaps   : {OVERLAP_PCTS}%")
    logger.info(f"  GPUs       : {gpus}  →  {n_workers} parallel worker(s)")
    logger.info("=" * 60)

    def _worker(cfg: tuple[str, int, int]) -> dict:
        mod, window, overlap_pct = cfg
        gpu = gpu_q.get()
        return _run_experiment(mod, window, overlap_pct, gpu, gpu_q)

    results: list[dict] = []
    t_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(_worker, cfg): cfg for cfg in configs}
        for future in concurrent.futures.as_completed(future_map):
            mod, w, o = future_map[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as exc:
                tag = f"sensor_{mod}_W{w}_S{max(1, int(w * (1 - o / 100)))}"
                logger.error(f"[ERROR]  {tag}: {exc}")

    total_elapsed = time.monotonic() - t_start
    logger.info(f"\nSweep complete in {total_elapsed / 60:.1f} min — "
                f"{len(results)}/{n_total} succeeded")
    _print_summary(results)


if __name__ == "__main__":
    main()
