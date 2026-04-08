"""LSTM LOSO training script – TensorFlow/Keras version.

Converted from Copy_of_Glasses_LSTM_80_20.ipynb.
Data loading and LOSO strategy mirror Glasses_LSTM_LOSO.py (PyTorch).
All stdout/stderr output is redirected to a run-specific .log file.
"""

import argparse
import glob
import os
import sys
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers import Adam


# ── Tee: write to log file AND terminal simultaneously ─────────────────────────
class _Tee:
    """Mirrors writes to two streams (log file + original terminal)."""

    def __init__(self, file_stream, terminal_stream):
        self._file = file_stream
        self._term = terminal_stream

    def write(self, data: str):
        self._file.write(data)
        self._term.write(data)

    def flush(self):
        self._file.flush()
        self._term.flush()

    # Make it behave as a proper stream
    def fileno(self):
        return self._term.fileno()


# ── Logging helpers ────────────────────────────────────────────────────────────
def log(msg: str = "") -> None:
    """Print a timestamped line to current stdout (tee'd to log file + terminal)."""
    if msg == "":
        print("", flush=True)
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


class EpochLogger(Callback):
    """Replaces Keras verbose=1 progress bars with clean timestamped lines."""

    def __init__(self, log_every: int = 10):
        super().__init__()
        self.log_every = log_every

    def on_epoch_end(self, epoch: int, logs=None):
        logs = logs or {}
        if (epoch + 1) % self.log_every == 0 or epoch == 0:
            log(
                f"  epoch {epoch + 1:4d} | "
                f"loss {logs.get('loss', 0):.4f}  acc {logs.get('accuracy', 0):.4f} | "
                f"val_loss {logs.get('val_loss', 0):.4f}  val_acc {logs.get('val_accuracy', 0):.4f}"
            )

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))


def _find_project_root(start_dir: str) -> str:
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

# ── Hyper-parameters (kept from notebook) ─────────────────────────────────────
RANDOM_SEED = 42
N_FEATURES = 6
N_CLASSES = 11
N_EPOCHS = 400
BATCH_SIZE = 128
LEARNING_RATE = 0.01

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

tf.random.set_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ── Data loading ───────────────────────────────────────────────────────────────
def _exp_no(exp_id: int) -> int:
    """Map raw experiment number to activity label 1-11."""
    return exp_id % 11 or 11


def load_windows(window_size: int, step_size: int, dataset_path: str = DATASET_PATH):
    """
    Returns
    -------
    X           : (n_windows, window_size, N_FEATURES) float32
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

    log(f"Loaded {X.shape[0]} windows | window={window_size} step={step_size}")
    log(f"Experiments: {meta['expID'].nunique()} | Subjects: {len(np.unique(subject_ids))}")
    log(f"Skipped (missing sensor): {skipped}")
    return X, y, subject_ids


# ── Model (kept from notebook) ─────────────────────────────────────────────────
def build_model(window_size: int, n_features: int, n_classes: int) -> Sequential:
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(window_size, n_features)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(128))
    model.add(Dropout(0.5))
    model.add(Dense(units=64, activation="relu"))
    model.add(Dense(n_classes, activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=LEARNING_RATE),
        metrics=["accuracy"],
    )
    return model


# ── LOSO run ───────────────────────────────────────────────────────────────────
def run(window_size: int, step_size: int, epochs: int):
    log(f"TF GPU devices: {tf.config.list_physical_devices('GPU')}")

    X, y, subject_ids = load_windows(window_size, step_size)
    log(f"X: {X.shape}  y: {y.shape}  subjects: {np.unique(subject_ids)}")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    unique_subjects = np.unique(subject_ids)
    fold_accuracies = []
    all_y_true = []
    all_y_pred = []
    best_acc = -1.0

    tag = f"W{window_size}_S{step_size}"
    model_path = os.path.join(MODELS_DIR, f"glasses_lstm_loso_tf_best_{tag}.keras")

    _, W, C = X.shape

    for subject in unique_subjects:
        test_mask = subject_ids == subject
        train_mask = ~test_mask

        X_train_raw = X[train_mask]
        X_test_raw = X[test_mask]
        y_train_enc = y_enc[train_mask]
        y_test_enc = y_enc[test_mask]

        scaler = StandardScaler()
        n_tr = X_train_raw.shape[0]
        X_train_sc = scaler.fit_transform(X_train_raw.reshape(-1, C)).reshape(n_tr, W, C).astype(np.float32)
        X_test_sc = scaler.transform(X_test_raw.reshape(-1, C)).reshape(X_test_raw.shape[0], W, C).astype(np.float32)

        # One-hot encode labels
        y_train_oh = np.asarray(pd.get_dummies(y_train_enc), dtype=np.float32)
        y_test_oh = np.asarray(pd.get_dummies(y_test_enc), dtype=np.float32)

        log("-" * 60)
        log(
            f"LOSO fold - held-out: User{subject} "
            f"(test={test_mask.sum()} train={train_mask.sum()})"
        )

        model = build_model(W, C, n_classes)

        fold_ckpt = os.path.join(MODELS_DIR, f"glasses_lstm_loso_tf_fold{subject}_{tag}.keras")
        mc = ModelCheckpoint(fold_ckpt, monitor="val_accuracy", mode="max", save_best_only=True, verbose=0)

        history = model.fit(
            X_train_sc, y_train_oh,
            epochs=epochs,
            validation_data=(X_test_sc, y_test_oh),
            batch_size=BATCH_SIZE,
            callbacks=[mc, EpochLogger(log_every=10)],
            verbose=0,
        )

        model = keras.models.load_model(fold_ckpt)
        predictions = model.predict(X_test_sc, batch_size=BATCH_SIZE, verbose=0)
        y_pred = np.argmax(predictions, axis=1)

        acc = accuracy_score(y_test_enc, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test_enc.tolist())
        all_y_pred.extend(y_pred.tolist())
        log(f"  User{subject} accuracy: {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            model.save(model_path)
            log(f"  New best model saved (acc={acc:.4f}) -> {model_path}")

        # Loss/accuracy curves per fold
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(history.history["loss"], "r--", label="Train loss")
        axes[0].plot(history.history["val_loss"], "g-", label="Val loss")
        axes[0].set_title(f"Loss - User{subject}")
        axes[0].set_xlabel("Epoch")
        axes[0].legend()
        axes[0].set_ylim(0)

        axes[1].plot(history.history["accuracy"], "r--", label="Train acc")
        axes[1].plot(history.history["val_accuracy"], "g-", label="Val acc")
        axes[1].set_title(f"Accuracy - User{subject}")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()
        axes[1].set_ylim(0)

        plt.tight_layout()
        fold_fig = os.path.join(FIGURES_DIR, f"lstm_tf_history_user{subject}_{tag}.png")
        plt.savefig(fold_fig, dpi=150)
        plt.close()

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    log("=" * 60)
    log("LOSO SUMMARY")
    log("=" * 60)
    for subject, acc in zip(unique_subjects, fold_accuracies):
        log(f"  User{subject}: {acc:.4f}")
    log(f"  Mean accuracy : {np.mean(fold_accuracies):.4f}")
    log(f"  Std  accuracy : {np.std(fold_accuracies):.4f}")
    log()

    class_names = [ACTIVITY_NAMES[le.classes_[i]] for i in range(len(le.classes_))]
    log("=" * 60)
    log("CLASSIFICATION REPORT (all folds combined)")
    log("=" * 60)
    log(classification_report(all_y_true, all_y_pred, target_names=class_names))

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
    plt.title(f"LSTM TF - Confusion Matrix (LOSO) [{tag}]", fontsize=14)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"lstm_tf_confusion_matrix_{tag}.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    log(f"Saved: {cm_path}")

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
    plt.title(f"LSTM TF - Per-subject LOSO Accuracy [{tag}]")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, f"lstm_tf_per_subject_accuracy_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    plt.close()
    log(f"Saved: {bar_path}")


# ── Entry point ────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="LSTM LOSO TF runner for Dataset_1A")
    parser.add_argument("--window", type=int, default=500, help="Sliding window size")
    parser.add_argument("--step", type=int, default=250, help="Sliding window step")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS, help="Training epochs per fold")
    return parser.parse_args()


def main():
    args = parse_args()
    tag = f"W{args.window}_S{args.step}"
    log_path = os.path.join(LOGS_DIR, f"glasses_lstm_loso_tf_{tag}.log")

    with open(log_path, "w", encoding="utf-8") as log_file:
        tee_out = _Tee(log_file, sys.__stdout__)
        tee_err = _Tee(log_file, sys.__stderr__)
        sys.stdout = tee_out
        sys.stderr = tee_err
        try:
            log("Starting run")
            log(f"Log file: {log_path}")
            log(f"Model directory: {MODELS_DIR}")
            run(window_size=args.window, step_size=args.step, epochs=args.epochs)
            log("Run finished")
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__


if __name__ == "__main__":
    main()
