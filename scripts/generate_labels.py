"""
Scan DataSets/ for all CSV files and generate a label index.

Output: scripts/exports/labels.csv
Columns: file_name, relative_path, exp_id, activity_id, user_id, sensor_name

Activity ID derivation:
  Dataset_1A / 1B : exp_id % 11  (or 11 if remainder is 0)  — 11 activities
  Dataset_2       : exp_id % 6   (or 6  if remainder is 0)  — 6 TUG activities

Usage:
  python scripts/generate_labels.py
  python scripts/generate_labels.py --data_dir DataSets --out scripts/exports/labels.csv
"""

import argparse
import glob
import os
import sys

import pandas as pd

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_ROOT     = os.path.join(_HERE, "..")
DATA_DIR  = os.path.join(_ROOT, "DataSets")
OUT_PATH  = os.path.join(_HERE, "exports", "labels.csv")

# ─── Activity mappings ────────────────────────────────────────────────────────

ACTIVITY_NAMES_11 = {
    1:  "Sitting – Reading a Book",
    2:  "Sitting – Writing on a Notebook",
    3:  "Using Computer – Typing",
    4:  "Using Computer – Browsing",
    5:  "Sitting – Moving head/body",
    6:  "Sitting – Moving chair",
    7:  "Sitting - Stand up from sitting",
    8:  "Standing",
    9:  "Walking",
    10: "Running",
    11: "Taking stairs",
}

ACTIVITY_NAMES_15 = {
    1:  "Sitting – Reading a Book",
    2:  "Sitting – Writing on a Notebook",
    3:  "Using Computer – Typing",
    4:  "Using Computer – Browsing",
    5:  "Sitting – Moving head/body",
    6:  "Sitting – Moving chair",
    7:  "Sitting - Stand up from sitting",
    8:  "Standing",
    9:  "Walking",
    10: "Running",
    11: "Taking stairs",
    12: "Sitting - Stationary > Wear it > Put it back > Stationary (15s delay between each step)",
    13: "Standing - Stationary > Wear it > Put it back > Stationary (15s delay between each step)",
    14: "Sitting - Pick item from floor",
    15: "Standing - Pick item from floor",
}

ACTIVITY_NAMES_6 = {
    1: "Sitting - Reading a Book",
    2: "Sitting - Using a Computer",
    3: "Sitting - Stand up from sitting",
    4: "Walking",
    5: "Standing - Pick items from the floor",
    6: "TUG - Sit > Stand up > Walk to a point > Turn > Walk back to chair > Sit down",
}

DATASET_CONFIG = {
    "Dataset_1A": {"mod": 11, "names": ACTIVITY_NAMES_11},
    "Dataset_1B": {"mod": 15, "names": ACTIVITY_NAMES_15},
    "Dataset_2":  {"mod": 6,  "names": ACTIVITY_NAMES_6},
}


def activity_id(exp_id: int, mod: int) -> int:
    return exp_id % mod or mod


# ─── Main ─────────────────────────────────────────────────────────────────────

def build_label_index(data_dir: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    if not csv_files:
        sys.exit(f"No CSV files found under: {data_dir}")

    rows = []
    skipped = 0

    for path in sorted(csv_files):
        fname  = os.path.basename(path)
        parts  = fname.split("_")

        # Expect: {exp_id}_MetaWear_..._..._SensorName_...csv
        if len(parts) < 5:
            skipped += 1
            continue

        try:
            exp_id_val = int(parts[0])
        except ValueError:
            skipped += 1
            continue

        sensor_name = parts[4]

        # user_id from parent folder (e.g. "User3" → 3)
        user_folder = os.path.basename(os.path.dirname(path))
        try:
            user_id_val = int(user_folder.replace("User", ""))
        except ValueError:
            skipped += 1
            continue

        # dataset — walk up until we find a known Dataset_* folder
        dataset = "Unknown"
        parts_path = path.split(os.sep)
        for part in parts_path:
            if part in DATASET_CONFIG:
                dataset = part
                break
        cfg     = DATASET_CONFIG.get(dataset, {"mod": 11, "names": ACTIVITY_NAMES_11})
        mod     = cfg["mod"]

        rel_path   = os.path.relpath(path, _ROOT)
        act_id     = activity_id(exp_id_val, mod)
        act_name   = cfg["names"].get(act_id, f"Activity_{act_id}")

        rows.append({
            "file_name":     fname,
            "relative_path": rel_path,
            "dataset":       dataset,
            "exp_id":        exp_id_val,
            "activity_id":   act_id,
            "activity_name": act_name,
            "user_id":       user_id_val,
            "sensor_name":   sensor_name,
        })

    df = pd.DataFrame(rows, columns=[
        "file_name", "relative_path", "dataset", "exp_id", "activity_id", "activity_name", "user_id", "sensor_name"
    ])

    print(f"Found   : {len(df):,} files")
    if skipped:
        print(f"Skipped : {skipped} (unrecognised filename pattern)")
    print(f"Users   : {df['user_id'].nunique()}")
    print(f"Sensors : {sorted(df['sensor_name'].unique())}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate CSV label index for DataSets/.")
    parser.add_argument("--data_dir", default=DATA_DIR,
                        help=f"Root dataset directory (default: {DATA_DIR})")
    parser.add_argument("--out", default=OUT_PATH,
                        help=f"Output CSV path (default: {OUT_PATH})")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = build_label_index(args.data_dir)
    df.to_csv(args.out, index=False)
    print(f"Saved   : {args.out}")


if __name__ == "__main__":
    main()
