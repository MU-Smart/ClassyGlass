"""
Ablation sweep — window sizes × overlap fractions × models, dispatched in
parallel across all available CUDA GPUs (falls back to CPU/MPS if none found).

Window sizes : 100, 200, 300, 400, 500  (samples)
Overlaps     : 0%, 25%, 50%, 60%        (step = window × (1 - overlap))
Models       : cnn, deep_conv_lstm

Each run.py call writes its own log to logs/<model>_W<w>_S<s>.log as usual.
A master sweep log is written to logs/ablation_<timestamp>.log.

Usage:
  python run_ablation.py
  python run_ablation.py --models cnn           # restrict to one model
  python run_ablation.py --workers 2            # override parallelism
"""

import argparse
import concurrent.futures
import itertools
import logging
import os
import queue
import subprocess
import sys
import time
from datetime import datetime

import torch

# ─── Paths ────────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR  = os.path.join(_HERE, "logs")
RUN_PY    = os.path.join(_HERE, "run.py")
os.makedirs(LOGS_DIR, exist_ok=True)

# ─── Sweep grid ───────────────────────────────────────────────────────────────

WINDOWS       = [200, 300, 500]
OVERLAP_PCTS  = [0, 25, 50]           # percent
ALL_MODELS    = ["deep_conv_lstm", "lstm"]

# ─── Logger ───────────────────────────────────────────────────────────────────

def _setup_logger(timestamp: str) -> logging.Logger:
    log = logging.getLogger("ablation")
    log.setLevel(logging.DEBUG)
    log.propagate = False

    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[union-attr]

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)

    log_path = os.path.join(LOGS_DIR, f"ablation_{timestamp}.log")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.info(f"Master log: {log_path}")
    return log


# ─── GPU detection ────────────────────────────────────────────────────────────

def get_available_gpus(log: logging.Logger) -> list[int]:
    if torch.cuda.is_available():
        n    = torch.cuda.device_count()
        gpus = list(range(n))
        log.info(f"Found {n} CUDA GPU(s):")
        for i in gpus:
            mem_gib = torch.cuda.get_device_properties(i).total_memory / 2**30
            log.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}  ({mem_gib:.1f} GiB)")
        return gpus

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.warning("No CUDA GPUs — using MPS (experiments serialised).")
        return [0]

    log.warning("No CUDA / MPS — using CPU (experiments serialised).")
    return [0]


# ─── Experiment worker ────────────────────────────────────────────────────────

def run_experiment(
    model: str,
    window: int,
    overlap_pct: int,
    gpu: int,
    log: logging.Logger,
) -> tuple[str, int]:
    step        = max(1, int(window * (1 - overlap_pct / 100)))
    tag         = f"{model}_W{window}_O{overlap_pct}"
    cmd         = [
        sys.executable, RUN_PY,
        "--model",  model,
        "--window", str(window),
        "--step",   str(step),
        "--gpu",    str(gpu),
    ]

    log.info(f"[START]  {tag:<32s}  gpu={gpu}  step={step}")
    t0   = time.monotonic()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.monotonic() - t0

    # Surface any stderr lines in the master log for easy debugging
    if proc.stderr.strip():
        for line in proc.stderr.strip().splitlines():
            log.warning(f"[STDERR] {tag}: {line}")

    status = "OK" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
    log.info(f"[DONE]   {tag:<32s}  {status}  ({elapsed / 60:.1f} min)")
    return tag, proc.returncode


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation sweep over window / overlap / model.")
    p.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS,
        help="Models to include (default: all)",
    )
    p.add_argument(
        "--workers", type=int, default=0,
        help="Override number of parallel workers (default: number of GPUs)",
    )
    return p.parse_args()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args      = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log       = _setup_logger(timestamp)

    gpus   = get_available_gpus(log)
    n_gpus = len(gpus)
    n_workers = args.workers if args.workers > 0 else n_gpus

    # Build GPU round-robin queue
    gpu_q: queue.Queue[int] = queue.Queue()
    for g in gpus:
        gpu_q.put(g)

    configs = list(itertools.product(args.models, WINDOWS, OVERLAP_PCTS))
    n_total = len(configs)

    log.info("=" * 60)
    log.info(f"Ablation sweep: {n_total} experiments")
    log.info(f"  Models   : {args.models}")
    log.info(f"  Windows  : {WINDOWS}")
    log.info(f"  Overlaps : {OVERLAP_PCTS}%")
    log.info(f"  GPUs     : {gpus}  →  {n_workers} parallel worker(s)")
    log.info("=" * 60)

    def _worker(cfg: tuple[str, int, int]) -> tuple[str, int]:
        model, window, overlap_pct = cfg
        gpu = gpu_q.get()
        try:
            return run_experiment(model, window, overlap_pct, gpu, log)
        finally:
            gpu_q.put(gpu)

    results: dict[str, int] = {}
    t_start = time.monotonic()

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_map = {pool.submit(_worker, cfg): cfg for cfg in configs}
        for future in concurrent.futures.as_completed(future_map):
            try:
                tag, rc = future.result()
            except Exception as exc:
                m, w, o = future_map[future]
                tag = f"{m}_W{w}_O{o}"
                log.error(f"[ERROR]  {tag}: {exc}")
                rc = -1
            results[tag] = rc

    total_elapsed = time.monotonic() - t_start
    ok     = [t for t, rc in results.items() if rc == 0]
    failed = [t for t, rc in results.items() if rc != 0]

    log.info("=" * 60)
    log.info(f"Sweep complete in {total_elapsed / 60:.1f} min — "
             f"{len(ok)}/{n_total} succeeded")
    if failed:
        log.info(f"{len(failed)} failed experiment(s):")
        for t in sorted(failed):
            log.info(f"  FAILED  {t}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
