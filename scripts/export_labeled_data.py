"""
Export all users' sensor data labeled with experiment/activity metadata to CSV.

Walks all three datasets (Dataset_1A, Dataset_1B, Dataset_2) inside
Processed-DataSets/, reads every sensor CSV, attaches metadata columns,
and saves one combined CSV per dataset plus an all-datasets CSV.

Output files (written to scripts/exports/):
  labeled_Dataset_1A.csv
  labeled_Dataset_1B.csv
  labeled_Dataset_2.csv
  labeled_all.csv

Each output row = one sensor sample with columns:
  dataset, user_id, experiment_id, activity_label, activity,
  sensor_type, epoch_ms, elapsed_s, <sensor axes...>

Usage:
  python scripts/export_labeled_data.py
  python scripts/export_labeled_data.py --source Datasets          # raw (unprocessed)
  python scripts/export_labeled_data.py --datasets 1A 1B           # subset
  python scripts/export_labeled_data.py --sensors Accelerometer Gyroscope
"""

import argparse
import glob
import os
import sys

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.join(_HERE, "..")
EXPORTS_DIR  = os.path.join(_HERE, "exports")

# ─── Activity mappings ────────────────────────────────────────────────────────

# Dataset_1A and Dataset_1B: experiment numbers cycle modulo 11
ACTIVITY_NAMES_11 = {
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

# Dataset_2: TUG test — 6 discrete activities (experiment IDs 1-6 per user)
ACTIVITY_NAMES_2 = {
    1: "Sitting (pre-stand)",
    2: "Stand up",
    3: "Walk forward",
    4: "Turn around",
    5: "Walk back",
    6: "Sit down",
}

DATASET_CONFIGS = {
    "1A": {"activity_map": ACTIVITY_NAMES_11, "mod": 11},
    # "1B": {"activity_map": ACTIVITY_NAMES_11, "mod": 11},
    # "2":  {"activity_map": ACTIVITY_NAMES_2,  "mod": None},
}

# ─── Sensor column maps ───────────────────────────────────────────────────────

# Maps sensor type → list of value columns present in that sensor's CSV
SENSOR_COLS = {
    "Accelerometer": ["x-axis (g)",     "y-axis (g)",     "z-axis (g)"],
    "Gyroscope":     ["x-axis (deg/s)", "y-axis (deg/s)", "z-axis (deg/s)"],
    "Magnetometer":  ["x-axis (T)",     "y-axis (T)",     "z-axis (T)"],
    "Pressure":      ["pressure (Pa)"],
}

COMMON_COLS = ["epoch (ms)", "elapsed (s)"]


def _activity_label(exp_id: int, mod: int | None) -> int:
    if mod is None:
        return exp_id
    return exp_id % mod or mod


def _load_sensor_file(path: str, sensor_type: str) -> pd.DataFrame | None:
    """Read one sensor CSV; return None if required columns are missing."""
    value_cols = SENSOR_COLS.get(sensor_type)
    if value_cols is None:
        return None
    needed = COMMON_COLS + value_cols
    try:
        df = pd.read_csv(path, usecols=needed)
    except (ValueError, KeyError):
        return None
    df = df.rename(columns={"epoch (ms)": "epoch_ms", "elapsed (s)": "elapsed_s"})
    return df


def export_dataset(dataset_key: str, source_root: str,
                   sensors: list[str]) -> pd.DataFrame:
    """
    Load all users / experiments for one dataset.
    Returns a DataFrame with metadata + sensor values.
    """
    cfg          = DATASET_CONFIGS[dataset_key]
    activity_map = cfg["activity_map"]
    mod          = cfg["mod"]

    dataset_dir = os.path.join(source_root, f"Dataset_{dataset_key}")
    all_files   = glob.glob(os.path.join(dataset_dir, "*", "*.csv"))
    if not all_files:
        print(f"  [WARN] No CSV files found in: {dataset_dir}", file=sys.stderr)
        return pd.DataFrame()

    chunks: list[pd.DataFrame] = []
    skipped = 0

    for path in sorted(all_files):
        fname = os.path.basename(path)
        parts = fname.split("_")
        if len(parts) < 5:
            skipped += 1
            continue

        try:
            exp_id = int(parts[0])
        except ValueError:
            skipped += 1
            continue

        sensor_type = parts[4]
        if sensor_type not in sensors:
            continue

        user_folder = os.path.basename(os.path.dirname(path))
        try:
            user_id = int(user_folder.replace("User", ""))
        except ValueError:
            skipped += 1
            continue

        df = _load_sensor_file(path, sensor_type)
        if df is None or df.empty:
            skipped += 1
            continue

        label = _activity_label(exp_id, mod)
        df.insert(0, "dataset",        f"Dataset_{dataset_key}")
        df.insert(1, "user_id",        user_id)
        df.insert(2, "experiment_id",  exp_id)
        df.insert(3, "activity_label", label)
        df.insert(4, "activity",       activity_map.get(label, f"Activity_{label}"))
        df.insert(5, "sensor_type",    sensor_type)

        chunks.append(df)

    if not chunks:
        print(f"  [WARN] Dataset_{dataset_key}: no data loaded.", file=sys.stderr)
        return pd.DataFrame()

    combined = pd.concat(chunks, ignore_index=True)
    print(
        f"  Dataset_{dataset_key}: {len(combined):,} rows | "
        f"{combined['user_id'].nunique()} users | "
        f"{combined['experiment_id'].nunique()} experiments | "
        f"{combined['sensor_type'].nunique()} sensor type(s)"
        + (f" | {skipped} files skipped" if skipped else "")
    )
    return combined


def main():
    parser = argparse.ArgumentParser(
        description="Export labeled sensor data to CSV."
    )
    parser.add_argument(
        "--source", default="Processed-DataSets",
        help="Top-level data directory relative to repo root "
             "(default: Processed-DataSets)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASET_CONFIGS),
        choices=list(DATASET_CONFIGS),
        metavar="DATASET",
        help="Datasets to export (default: 1A 1B 2)"
    )
    parser.add_argument(
        "--sensors", nargs="+",
        default=list(SENSOR_COLS),
        choices=list(SENSOR_COLS),
        metavar="SENSOR",
        help="Sensor types to include (default: all)"
    )
    args = parser.parse_args()

    source_root = os.path.join(_ROOT, args.source)
    if not os.path.isdir(source_root):
        sys.exit(f"Source directory not found: {source_root}")

    os.makedirs(EXPORTS_DIR, exist_ok=True)

    print(f"Source : {source_root}")
    print(f"Output : {EXPORTS_DIR}")
    print(f"Sensors: {', '.join(args.sensors)}")
    print()

    all_parts: list[pd.DataFrame] = []

    for ds in args.datasets:
        print(f"Loading Dataset_{ds} …")
        df = export_dataset(ds, source_root, args.sensors)
        if df.empty:
            continue

        out_path = os.path.join(EXPORTS_DIR, f"labeled_Dataset_{ds}.csv")
        df.to_csv(out_path, index=False)
        print(f"  -> saved: {out_path}")
        all_parts.append(df)

    if all_parts:
        all_df = pd.concat(all_parts, ignore_index=True)
        out_all = os.path.join(EXPORTS_DIR, "labeled_all.csv")
        all_df.to_csv(out_all, index=False)
        print(f"\nCombined ({len(all_df):,} rows) -> {out_all}")


if __name__ == "__main__":
    main()
