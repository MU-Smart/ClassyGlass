"""
XGBoost with Leave-One-Subject-Out (LOSO) cross-validation.
Reads pre-extracted feature CSVs from DatasetsFeatureExtracted/Dataset_1A/300_150/.

Usage:
  python run_xgboost.py
  python run_xgboost.py --data_dir ../../DatasetsFeatureExtracted/Dataset_1A/300_150
  python run_xgboost.py --n_estimators 200 --max_depth 8
"""

import argparse
import glob
import logging
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_HERE, "..", "..", "DatasetsFeatureExtracted", "Dataset_1A", "500_250")
LOGS_DIR    = os.path.join(_HERE, "logs")
FIGURES_DIR = os.path.join(_HERE, "figures")

for _d in (LOGS_DIR, FIGURES_DIR):
    os.makedirs(_d, exist_ok=True)

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

FEATURE_COLS = [
    "acc_x_mean", "acc_x_var",
    "acc_y_mean", "acc_y_var",
    "acc_z_mean", "acc_z_var",
    "acc_sum_mean", "acc_abssum_mean", "acc_sum_var", "acc_abssum_var", "acc_maxabssum",
    "gyr_x_mean", "gyr_x_var",
    "gyr_y_mean", "gyr_y_var",
    "gyr_z_mean", "gyr_z_var",
    "gyr_sum_mean", "gyr_abssum_mean", "gyr_sum_var", "gyr_abssum_var", "gyr_maxabssum",
]

# ─── Logger ───────────────────────────────────────────────────────────────────

logger = logging.getLogger("xgboost_loso")
logger.setLevel(logging.DEBUG)
logger.propagate = False
_fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_features(data_dir: str):
    """
    Load all per-user CSV files, infer subject_id from filename.

    Returns
    -------
    X           : (n_windows, n_features)  float64
    y           : (n_windows,)  int  activity labels
    subject_ids : (n_windows,)  int
    """
    csv_files = sorted(glob.glob(os.path.join(data_dir, "User*_features_*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No feature CSVs found in: {data_dir}")

    frames = []
    for path in csv_files:
        fname = os.path.basename(path)
        # Filename: User<N>_features_W<W>_O<O>.csv
        try:
            subject_id = int(fname.split("_")[0].replace("User", ""))
        except ValueError:
            logger.warning(f"Could not parse subject ID from {fname}, skipping.")
            continue
        df = pd.read_csv(path)
        df["subject_id"] = subject_id
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    # Validate feature columns exist
    missing = [c for c in FEATURE_COLS if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns in CSV: {missing}")

    X           = data[FEATURE_COLS].values.astype(np.float64)
    y           = data["activity_id"].values.astype(int)
    subject_ids = data["subject_id"].values.astype(int)

    logger.info(
        f"Loaded {len(X)} windows | {len(np.unique(subject_ids))} subjects | "
        f"{len(np.unique(y))} classes | {X.shape[1]} features"
    )
    return X, y, subject_ids


# ─── LOSO cross-validation ────────────────────────────────────────────────────

def run_loso(X, y, subject_ids, xgb_params: dict):
    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    unique_subjects  = np.unique(subject_ids)
    fold_accuracies: list[float] = []
    all_y_true:      list[int]   = []
    all_y_pred:      list[int]   = []

    for subject in unique_subjects:
        test_mask  = subject_ids == subject
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y_enc[train_mask], y_enc[test_mask]

        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        logger.info("─" * 60)
        logger.info(
            f"LOSO fold — held-out: User{subject} "
            f"({test_mask.sum()} test windows, {train_mask.sum()} train windows)"
        )

        model = XGBClassifier(**xgb_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        logger.info(f"  User{subject} accuracy: {acc:.4f}")

    return fold_accuracies, np.array(all_y_true), np.array(all_y_pred), le


# ─── Hyperparameter tuning ────────────────────────────────────────────────────

PARAM_GRID = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 5, 6, 8, 10],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "gamma":            [0, 0.1, 0.3, 0.5],
}


def tune_params(X, y, base_params: dict, n_iter: int, n_folds: int) -> dict:
    """
    RandomizedSearchCV over PARAM_GRID. Returns best params merged with base_params
    (device / n_jobs / eval_metric are preserved from base_params).
    """
    logger.info("=" * 60)
    logger.info(f"Hyperparameter search — {n_iter} iterations, {n_folds}-fold CV")
    logger.info("=" * 60)

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    # Strip keys that clash with the search grid so base_params acts as fixed defaults
    fixed = {k: v for k, v in base_params.items() if k not in PARAM_GRID}

    estimator = XGBClassifier(**fixed)
    cv        = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    search    = RandomizedSearchCV(
        estimator,
        PARAM_GRID,
        n_iter      = n_iter,
        cv          = cv,
        scoring     = "accuracy",
        n_jobs      = base_params.get("n_jobs", -1),
        random_state= 42,
        verbose     = 1,
        refit       = False,
    )
    search.fit(X_sc, y_enc)

    best = search.best_params_
    logger.info(f"Best CV accuracy : {search.best_score_:.4f}")
    logger.info(f"Best params      : {best}")

    return {**base_params, **best}


# ─── Reporting ────────────────────────────────────────────────────────────────

def print_report(fold_accuracies, y_true, y_pred, le, subject_ids_unique, tag: str):
    class_names = [ACTIVITY_NAMES.get(le.classes_[i], str(le.classes_[i]))
                   for i in range(len(le.classes_))]

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
    plt.title(f"XGBoost – Confusion Matrix (LOSO)  [{tag}]")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, f"xgboost_confusion_matrix_{tag}.png")
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
    plt.title(f"XGBoost – Per-subject LOSO Accuracy  [{tag}]")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(FIGURES_DIR, f"xgboost_per_subject_accuracy_{tag}.png")
    plt.savefig(bar_path, dpi=150)
    logger.info(f"Per-subject accuracy chart saved to: {bar_path}")
    plt.close()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost LOSO on pre-extracted Dataset_1A features."
    )
    parser.add_argument(
        "--data_dir", default=DATA_DIR,
        help=f"Directory with per-user feature CSVs (default: {DATA_DIR})"
    )
    parser.add_argument("--n_estimators", type=int, default=300,
                        help="Number of XGBoost trees (default: 300)")
    parser.add_argument("--max_depth",    type=int, default=6,
                        help="Max tree depth (default: 6)")
    parser.add_argument("--learning_rate", type=float, default=0.1,
                        help="Learning rate / eta (default: 0.1)")
    parser.add_argument("--subsample",    type=float, default=0.8,
                        help="Row subsampling ratio (default: 0.8)")
    parser.add_argument("--colsample_bytree", type=float, default=0.8,
                        help="Column subsampling per tree (default: 0.8)")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="CPU threads, ignored when using CUDA (default: -1)")
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA device index (default: 0). Ignored if CUDA unavailable.")
    parser.add_argument("--tune", action="store_true",
                        help="Run RandomizedSearchCV before LOSO to find best hyperparameters.")
    parser.add_argument("--tune_iter", type=int, default=30,
                        help="Number of random search iterations (default: 30)")
    parser.add_argument("--tune_folds", type=int, default=5,
                        help="K for stratified k-fold during tuning (default: 5)")
    return parser.parse_args()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(_fmt)
    logger.addHandler(_console_handler)

    args = parse_args()
    tag  = f"xgb_n{args.n_estimators}_d{args.max_depth}_lr{args.learning_rate}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path  = os.path.join(LOGS_DIR, f"xgboost_{timestamp}.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(_fmt)
    logger.addHandler(fh)

    # XGBoost only supports CUDA for GPU acceleration; MPS is not supported.
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_idx = args.gpu if args.gpu < n_gpus else 0
        device = f"cuda:{gpu_idx}"
        n_jobs = 1  # not used in GPU mode
        logger.info(f"Device  : {device} ({torch.cuda.get_device_name(gpu_idx)})")
    else:
        device = "cpu"
        n_jobs = args.n_jobs
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.warning("MPS detected but XGBoost does not support Apple MPS — using CPU.")
        else:
            logger.info("CUDA not available — using CPU.")

    xgb_params = dict(
        n_estimators     = args.n_estimators,
        max_depth        = args.max_depth,
        learning_rate    = args.learning_rate,
        subsample        = args.subsample,
        colsample_bytree = args.colsample_bytree,
        eval_metric      = "mlogloss",
        random_state     = 42,
        device           = device,
        n_jobs           = n_jobs,
    )

    logger.info("=" * 60)
    logger.info("XGBoost LOSO — Dataset_1A  300_150")
    logger.info(f"Data    : {args.data_dir}")
    logger.info(f"Log     : {log_path}")
    logger.info("=" * 60)

    X, y, subject_ids = load_features(args.data_dir)

    if args.tune:
        xgb_params = tune_params(X, y, xgb_params, args.tune_iter, args.tune_folds)
        tag = tag + "_tuned"

    logger.info("=" * 60)
    logger.info(f"LOSO params : {xgb_params}")
    logger.info("=" * 60)

    fold_accuracies, y_true, y_pred, le = run_loso(X, y, subject_ids, xgb_params)
    print_report(fold_accuracies, y_true, y_pred, le, np.unique(subject_ids), tag)

    logger.info("=" * 60)
    logger.info(f"XGBoost LOSO complete  |  mean acc = {np.mean(fold_accuracies):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
