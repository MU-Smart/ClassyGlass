"""
Preprocessor script: reads all files from Datasets/, applies signal preprocessing
to CSVs with x/y/z axes (Accelerometer, Gyroscope, Magnetometer), and saves results
to Proccessed-DataSets/ mirroring the original directory structure.
Non-XYZ CSVs and non-CSV files are copied as-is.
"""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, sosfiltfilt


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _hampel_filter_vectorized(signal: np.ndarray, window_size: int = 10, n_sigmas: float = 3) -> np.ndarray:
    """
    Fully vectorized Hampel filter — significantly faster for large arrays.
    Uses stride tricks to build a sliding window matrix in one operation.
    """
    if not isinstance(signal, np.ndarray):
        signal = np.asarray(signal, dtype=float)

    L = len(signal)
    new_signal = signal.copy()

    win_full = 2 * window_size + 1
    shape = (L - 2 * window_size, win_full)
    strides = (signal.strides[0], signal.strides[0])
    windows = np.lib.stride_tricks.as_strided(signal, shape=shape, strides=strides)

    medians = np.median(windows, axis=1)
    mads = stats.median_abs_deviation(windows, axis=1)

    threshold = n_sigmas * np.maximum(mads, 1e-10)
    center_vals = signal[window_size : L - window_size]
    outlier_mask = np.abs(center_vals - medians) > threshold

    new_signal[window_size : L - window_size][outlier_mask] = medians[outlier_mask]

    return new_signal


def _run_hampel(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Hampel filter to valueX, valueY, valueZ columns."""
    result = df.copy()
    for col in ("valueX", "valueY", "valueZ"):
        result[col] = _hampel_filter_vectorized(df[col].values, window_size=10, n_sigmas=3)
    return result


# Butterworth filter design using SOS for stability
def _design_butterworth_filter(cutoff, fs, btype, order=4):
    """Design a Butterworth filter and return SOS coefficients."""
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    return butter(order, normal_cutoff, btype=btype, output='sos')


def run_butterworth(data, fs=100, highpass_cutoff=0.3, lowpass_cutoff=25, order=4):
    """
    Apply bandpass filtering (highpass + lowpass) to X, Y, Z accelerometer columns.
    Uses SOS format for numerical stability.
    """
    # Design filters once, reuse across axes
    sos_high = _design_butterworth_filter(highpass_cutoff, fs, btype='high', order=order)
    sos_low  = _design_butterworth_filter(lowpass_cutoff,  fs, btype='low',  order=order)

    cols = ['valueX', 'valueY', 'valueZ']
    raw = data[cols].values  # (N, 3) — process all axes at once

    filtered = sosfiltfilt(sos_high, raw, axis=0)
    filtered = sosfiltfilt(sos_low,  filtered, axis=0)

    data_filtered = data.copy()
    data_filtered[cols] = filtered

    return data_filtered


# ---------------------------------------------------------------------------
# CSV column detection
# ---------------------------------------------------------------------------

# Columns that contain x/y/z axes (vary by sensor unit suffix)
_XYZ_PREFIXES = ("x-axis", "y-axis", "z-axis")


def _xyz_columns(df: pd.DataFrame) -> tuple[str, str, str] | None:
    """
    Return (x_col, y_col, z_col) if the DataFrame has x/y/z axis columns,
    otherwise return None.
    """
    x = y = z = None
    for col in df.columns:
        col_lower = col.lower()
        if col_lower.startswith("x-axis"):
            x = col
        elif col_lower.startswith("y-axis"):
            y = col
        elif col_lower.startswith("z-axis"):
            z = col
    if x and y and z:
        return x, y, z
    return None


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_csv(src: Path, dst: Path) -> None:
    """Load a CSV, apply preprocessing if it has XYZ axes, then save."""
    df = pd.read_csv(src)
    cols = _xyz_columns(df)

    if cols is None:
        # No XYZ axes — copy the CSV unchanged
        shutil.copy2(src, dst)
        return

    x_col, y_col, z_col = cols

    # Rename to standard names expected by the processing functions
    df = df.rename(columns={x_col: "valueX", y_col: "valueY", z_col: "valueZ"})

    df = _run_hampel(df)
    df = run_butterworth(df)

    # Restore original column names before saving
    df = df.rename(columns={"valueX": x_col, "valueY": y_col, "valueZ": z_col})

    df.to_csv(dst, index=False)


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def preprocess_datasets(
    src_root: Path,
    dst_root: Path,
) -> None:
    """
    Walk src_root recursively. For each file:
      - CSV with x/y/z axes → apply Hampel + remove_stalled, save to dst_root
      - CSV without x/y/z   → copy to dst_root
      - Any other file       → copy to dst_root
    Directory structure under src_root is mirrored in dst_root.
    """
    files = list(src_root.rglob("*"))
    total = sum(1 for f in files if f.is_file())
    processed = 0

    for src_path in files:
        if not src_path.is_file():
            continue

        rel = src_path.relative_to(src_root)
        dst_path = dst_root / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        processed += 1
        suffix = src_path.suffix.lower()
        print(f"[{processed}/{total}] {rel}", end=" ... ", flush=True)

        if suffix == ".csv":
            try:
                _process_csv(src_path, dst_path)
                print("done")
            except Exception as exc:
                print(f"ERROR: {exc}")
                shutil.copy2(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
            print("copied")

    print(f"\nFinished: {processed} file(s) processed → {dst_root}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    src_root = repo_root / "Datasets"
    dst_root = repo_root / "Processed-DataSets"

    if not src_root.exists():
        raise SystemExit(f"Source directory not found: {src_root}")

    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"Source : {src_root}")
    print(f"Output : {dst_root}\n")

    preprocess_datasets(src_root, dst_root)
