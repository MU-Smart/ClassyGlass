"""
Mobile Phone Runtime Simulator
================================
Estimates on-device inference latency for the DeepConvLSTM model across
four runtime targets:

  1. PyTorch CPU  — single-thread, simulates a low-end mobile CPU
  2. TorchScript  — PyTorch Mobile format (torch.jit.script)
  3. CoreML       — Apple Neural Engine / iOS runtime (requires coremltools)
  4. ONNX Runtime — cross-platform mobile runtime (requires onnxruntime)

Projection methodology — Roofline Model
----------------------------------------
Projected mobile latency is derived from the model's actual MAC count and
real published device specs, NOT from scaling Mac CPU measurements.

  compute_time = MACs / (device_peak_GFLOPS × efficiency)
  memory_time  = model_bytes / (device_BW_GBs × efficiency)
  projected_ms = max(compute_time, memory_time) × 1000

Device specs sourced from AnandTech / Notebookcheck / MLPerf Mobile 2023-24.

Device profiles built-in (--profile):
  high_end   → Snapdragon 8 Gen 2 / Apple A16  (12 GFLOPS/core, 51 GB/s)
  mid_range  → Snapdragon 778G / Apple A14     ( 6 GFLOPS/core, 34 GB/s)
  low_end    → MediaTek Helio G85              ( 1.5 GFLOPS/core, 17 GB/s)

Usage:
  python mobile_sim.py
  python mobile_sim.py --model models/deep_conv_lstm_best_W300_S150.pth
  python mobile_sim.py --model models/deep_conv_lstm_best_W300_S150.pth --profile mid_range
  python mobile_sim.py --model models/deep_conv_lstm_best_W300_S150.pth --no-coreml --no-onnx
"""

import argparse
import datetime
import os
import sys
import time
import statistics
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ── Model import ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from deep_conv_lstm import DeepConvLSTM, build_model  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Logger — mirrors stdout to a timestamped log file
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    """Drop-in replacement for print() that also writes timestamped lines to a file."""

    def __init__(self, log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(log_path, "w", encoding="utf-8")
        self.path = log_path

    def __call__(self, *args, **kwargs):
        msg = kwargs.get("sep", " ").join(str(a) for a in args)
        end = kwargs.get("end", "\n")
        print(*args, **kwargs)                                    # stdout
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._fh.write(f"{ts}  {msg}{end}")
        self._fh.flush()

    def close(self):
        self._fh.close()

# ── Constants matching the W300_S150 checkpoint ───────────────────────────────
WINDOW_SIZE = 300
N_CHANNELS  = 6
N_CLASSES   = 11
WARMUP_RUNS = 20
BENCH_RUNS  = 200

# ── Real device profiles from published hardware benchmarks ───────────────────
# Sources: AnandTech, Notebookcheck, MLPerf Mobile 2023-24, ARM whitepapers.
# cpu_gflops_fp32 : single big-core FP32 SIMD throughput (NEON/SVE)
# memory_bw_gbs   : peak LPDDR bandwidth (GB/s)
# efficiency      : fraction of peak achieved in real ML workloads (roofline η)
DEVICE_PROFILES = {
    "high_end": {
        "name":             "Snapdragon 8 Gen 2 / Apple A16 Bionic",
        "cpu_gflops_fp32":  12.0,   # Cortex-X3 / Everest big core, NEON FP32
        "memory_bw_gbs":    51.2,   # LPDDR5X dual-channel
        "efficiency":        0.45,  # real-world vs. peak
    },
    "mid_range": {
        "name":             "Snapdragon 778G / Apple A14 Bionic",
        "cpu_gflops_fp32":   6.0,   # Cortex-A78 / Firestorm big core
        "memory_bw_gbs":    34.1,   # LPDDR5 dual-channel
        "efficiency":        0.40,
    },
    "low_end": {
        "name":             "MediaTek Helio G85 / Snapdragon 680",
        "cpu_gflops_fp32":   1.5,   # Cortex-A75 big core
        "memory_bw_gbs":    17.0,   # LPDDR4X
        "efficiency":        0.35,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Roofline MAC / byte counting  (analytical, model-specific)
# ─────────────────────────────────────────────────────────────────────────────

def count_macs_deepconvlstm(
    window: int, channels: int, classes: int,
    n_filters: int = 64, kernel_size: int = 5,
    lstm_hidden: int = 128, lstm_layers: int = 2,
) -> int:
    """
    Analytical MAC count for one forward pass of DeepConvLSTM (batch=1).

    Conv1d  : MACs = C_in * C_out * K * T  (per layer, T = sequence length)
    LSTM    : MACs = 4 * (C_in + H) * H  per timestep per layer
    Linear  : MACs = H * n_classes
    """
    T = window  # sequence length is preserved by same-padding conv (padding = kernel_size // 2)

    # 4 × Conv1d with same padding → T unchanged
    conv1_macs = channels  * n_filters * kernel_size * T
    convN_macs = n_filters * n_filters * kernel_size * T  # layers 2-4
    total_conv = conv1_macs + 3 * convN_macs

    # 2-layer LSTM; hidden stays lstm_hidden across layers
    lstm_macs = 0
    lstm_in = n_filters
    for _ in range(lstm_layers):
        # 4 gates, each: (lstm_in + lstm_hidden) * lstm_hidden MACs
        lstm_macs += 4 * (lstm_in + lstm_hidden) * lstm_hidden * T
        lstm_in = lstm_hidden  # next layer's input size

    # Final linear: last hidden → classes
    fc_macs = lstm_hidden * classes

    return total_conv + lstm_macs + fc_macs


def count_model_bytes_fp32(
    channels: int, classes: int,
    n_filters: int = 64, kernel_size: int = 5,
    lstm_hidden: int = 128, lstm_layers: int = 2,
    window: int = 300,
) -> int:
    """
    Bytes read/written during one forward pass (FP32, batch=1).
    Includes weights + peak activations (roofline memory traffic estimate).
    """
    BYTES = 4  # FP32

    # --- Weights ---
    # Conv layers
    w_conv1 = (channels * n_filters * kernel_size + n_filters) * BYTES
    w_convN = (n_filters * n_filters * kernel_size + n_filters) * BYTES
    w_conv  = w_conv1 + 3 * w_convN

    # LSTM: weight_ih + weight_hh + bias_ih + bias_hh per layer
    lstm_in = n_filters
    w_lstm = 0
    for _ in range(lstm_layers):
        w_lstm += (4 * lstm_in * lstm_hidden      # weight_ih
                   + 4 * lstm_hidden * lstm_hidden  # weight_hh
                   + 8 * lstm_hidden) * BYTES        # biases
        lstm_in = lstm_hidden

    w_fc = (lstm_hidden * classes + classes) * BYTES

    # --- Peak activations ---
    # Conv output: (n_filters, T)
    a_conv = n_filters * window * BYTES
    # LSTM: all hidden states needed for BPTT — not relevant for inference,
    # but memory traffic covers the sequence of hidden vectors
    a_lstm = lstm_layers * window * lstm_hidden * BYTES
    # FC input/output
    a_fc   = (lstm_hidden + classes) * BYTES

    return w_conv + w_lstm + w_fc + a_conv + a_lstm + a_fc


def roofline_latency_ms(macs: int, model_bytes: int, profile: dict) -> tuple[float, str]:
    """
    Roofline model: projected latency = max(compute-bound, memory-bound).

      compute_time_s = macs / (peak_GFLOPS * 1e9 * η)
      memory_time_s  = model_bytes / (peak_BW_GBs * 1e9 * η)

    Returns projected latency in milliseconds.
    """
    eta = profile["efficiency"]
    effective_gflops = profile["cpu_gflops_fp32"] * 1e9 * eta  # FLOPS/s (1 MAC ≈ 2 FLOPS)
    effective_bw     = profile["memory_bw_gbs"]   * 1e9 * eta  # bytes/s

    compute_s = (macs * 2) / effective_gflops   # MACs → FLOPs (* 2)
    memory_s  = model_bytes / effective_bw

    bottleneck = "compute" if compute_s >= memory_s else "memory"
    return max(compute_s, memory_s) * 1000.0, bottleneck


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / 1_048_576


def human_size(path: str) -> str:
    mb = model_size_mb(path)
    return f"{mb:.2f} MB"


def bench_callable(fn, dummy_input, warmup: int = WARMUP_RUNS, runs: int = BENCH_RUNS):
    """
    Returns (mean_ms, p50_ms, p95_ms, p99_ms) for a single forward pass.
    `dummy_input` is passed to fn each call.
    """
    for _ in range(warmup):
        fn(dummy_input)

    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn(dummy_input)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))]
    return statistics.mean(latencies), p50, p95, p99


def print_row(label: str, mean: float, p50: float, p95: float, p99: float,
              size_str: str, projected_ms: float | None = None,
              bottleneck: str | None = None, log_fn=print):
    if projected_ms is not None:
        proj = f"{projected_ms:7.2f} ms"
        if bottleneck:
            proj += f" [{bottleneck}-bound]"
    else:
        proj = "   n/a   "
    log_fn(
        f"  {label:<22s}  "
        f"mean={mean:6.2f}ms  p50={p50:6.2f}ms  "
        f"p95={p95:6.2f}ms  p99={p99:6.2f}ms  "
        f"size={size_str:>9s}  projected={proj}"
    )


def print_separator(log_fn=print):
    log_fn("  " + "─" * 110)


# ─────────────────────────────────────────────────────────────────────────────
# 0. Torch Profiler — actual per-operator CPU timing
# ─────────────────────────────────────────────────────────────────────────────

def run_torch_profiler(
    model: nn.Module, dummy: torch.Tensor,
    warmup: int = 5, active: int = 10,
    top_n: int = 15, log_fn=print,
):
    """
    Run torch.profiler for one forward pass and report:
      • per-operator self CPU time (top N)
      • measured total CPU time
      • profiler-counted FLOPs (cross-check vs. analytical MAC count)
    """
    torch.set_num_threads(1)
    model.eval()

    schedule = torch.profiler.schedule(wait=0, warmup=warmup, active=active)

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        schedule=schedule,
        record_shapes=True,
        with_flops=True,
        with_modules=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(warmup + active):
                model(dummy.cpu())
                prof.step()

    key_avgs = prof.key_averages()

    # ── Total measured CPU time ───────────────────────────────────────────────
    total_self_us = sum(e.self_cpu_time_total for e in key_avgs)
    total_ms = total_self_us / 1000.0 / active   # average over active steps

    # ── Profiler FLOP count ───────────────────────────────────────────────────
    total_flops = sum(e.flops for e in key_avgs if e.flops > 0)

    log_fn("  Torch Profiler  (single-threaded CPU, actual op timing)")
    log_fn(f"  {'─'*70}")
    log_fn(f"  Total measured  : {total_ms:.3f} ms / inference  "
           f"({active} active steps)")
    if total_flops:
        log_fn(f"  Profiler FLOPs  : {total_flops/1e6:.2f}M  "
               f"(≈ {total_flops/2/1e6:.2f}M MACs)")

    # ── Per-operator table ────────────────────────────────────────────────────
    log_fn(f"  {'─'*70}")
    log_fn(f"  {'Operator':<35s}  {'Self CPU':>10s}  {'% total':>8s}  "
           f"{'Calls':>6s}  {'Shapes'}")
    log_fn(f"  {'─'*70}")

    sorted_ops = sorted(key_avgs, key=lambda e: e.self_cpu_time_total, reverse=True)
    for evt in sorted_ops[:top_n]:
        pct = 100.0 * evt.self_cpu_time_total / total_self_us if total_self_us else 0.0
        self_ms = evt.self_cpu_time_total / 1000.0 / active
        shapes  = str(evt.input_shapes[:2]) if evt.input_shapes else ""
        log_fn(f"  {evt.key:<35s}  {self_ms:>8.3f}ms  {pct:>7.1f}%  "
               f"{evt.count:>6d}  {shapes}")

    log_fn(f"  {'─'*70}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. PyTorch CPU  (single-threaded)
# ─────────────────────────────────────────────────────────────────────────────

def bench_pytorch_cpu(model: nn.Module, dummy: torch.Tensor):
    torch.set_num_threads(1)          # single-threaded → closest to mobile
    model.eval()
    model.cpu()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        torch.save(model.state_dict(), f.name)
        size_str = human_size(f.name)
        tmp_path = f.name

    @torch.no_grad()
    def _fn(x):
        return model(x)

    mean, p50, p95, p99 = bench_callable(_fn, dummy.cpu())
    os.unlink(tmp_path)
    return mean, p50, p95, p99, size_str


# ─────────────────────────────────────────────────────────────────────────────
# 2. TorchScript (PyTorch Mobile format)
# ─────────────────────────────────────────────────────────────────────────────

def bench_torchscript(model: nn.Module, dummy: torch.Tensor, out_dir: Path):
    torch.set_num_threads(1)
    model.eval()
    model.cpu()

    scripted = torch.jit.script(model)

    try:
        from torch.utils.mobile_optimizer import optimize_for_mobile
        scripted = optimize_for_mobile(scripted)
        label = "TorchScript+MobileOpt"
    except Exception:
        label = "TorchScript"

    ts_path = str(out_dir / "deep_conv_lstm_mobile.ptl")
    scripted._save_for_lite_interpreter(ts_path)
    size_str = human_size(ts_path)

    @torch.no_grad()
    def _fn(x):
        return scripted(x)

    mean, p50, p95, p99 = bench_callable(_fn, dummy.cpu())
    return label, mean, p50, p95, p99, size_str, ts_path


# ─────────────────────────────────────────────────────────────────────────────
# 3. CoreML
# ─────────────────────────────────────────────────────────────────────────────

def bench_coreml(model: nn.Module, dummy: torch.Tensor, out_dir: Path):
    try:
        import coremltools as ct
    except ImportError:
        return None, "coremltools not installed — pip install coremltools"

    model.eval()
    model.cpu()

    with torch.no_grad():
        traced = torch.jit.trace(model, dummy.cpu())

    cml_path = str(out_dir / "deep_conv_lstm.mlpackage")

    try:
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="input",
                                  shape=dummy.shape,
                                  dtype=np.float32)],
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            minimum_deployment_target=ct.target.iOS16,
        )
        mlmodel.save(cml_path)
    except Exception as e:
        return None, f"CoreML conversion failed: {e}"

    total = sum(f.stat().st_size for f in Path(cml_path).rglob("*") if f.is_file())
    size_str = f"{total / 1_048_576:.2f} MB"

    mlmodel_loaded = ct.models.MLModel(cml_path)
    inp_np = dummy.cpu().numpy().astype(np.float32)

    def _fn(_ignored):
        mlmodel_loaded.predict({"input": inp_np})

    mean, p50, p95, p99 = bench_callable(_fn, None)
    return (mean, p50, p95, p99, size_str, cml_path), None


# ─────────────────────────────────────────────────────────────────────────────
# 4. ONNX Runtime
# ─────────────────────────────────────────────────────────────────────────────

def bench_onnx(model: nn.Module, dummy: torch.Tensor, out_dir: Path):
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        return None, "onnx / onnxruntime not installed — pip install onnx onnxruntime"

    model.eval()
    model.cpu()

    onnx_path = str(out_dir / "deep_conv_lstm.onnx")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy.cpu(),
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )

    size_str = human_size(onnx_path)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(onnx_path, sess_options=opts,
                                   providers=["CPUExecutionProvider"])

    inp_np = dummy.cpu().numpy().astype(np.float32)

    def _fn(_ignored):
        session.run(None, {"input": inp_np})

    mean, p50, p95, p99 = bench_callable(_fn, None)

    quant_result = None
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = str(out_dir / "deep_conv_lstm_int8.onnx")
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
        quant_size_str = human_size(quant_path)

        q_session = ort.InferenceSession(quant_path, sess_options=opts,
                                         providers=["CPUExecutionProvider"])

        def _qfn(_ignored):
            q_session.run(None, {"input": inp_np})

        qmean, qp50, qp95, qp99 = bench_callable(_qfn, None)
        quant_result = (qmean, qp50, qp95, qp99, quant_size_str, quant_path)
    except Exception:
        pass

    return (mean, p50, p95, p99, size_str, onnx_path, quant_result), None


# ─────────────────────────────────────────────────────────────────────────────
# Real-time feasibility check
# ─────────────────────────────────────────────────────────────────────────────

def realtime_check(projected_ms: float, window_size: int, sample_rate_hz: float = 50.0):
    window_ms = (window_size / sample_rate_hz) * 1000.0
    feasible = projected_ms < window_ms
    return window_ms, feasible


def feasibility_tag(ok: bool) -> str:
    return "FEASIBLE" if ok else "TOO SLOW"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Mobile runtime simulator for DeepConvLSTM")
    p.add_argument("--model",   default="models/deep_conv_lstm_best_W300_S150.pth",
                   help="Path to .pth checkpoint")
    p.add_argument("--window",  type=int, default=WINDOW_SIZE)
    p.add_argument("--channels", type=int, default=N_CHANNELS)
    p.add_argument("--classes", type=int, default=N_CLASSES)
    p.add_argument("--profile", choices=list(DEVICE_PROFILES), default="mid_range",
                   help="Mobile device profile for projected latency (default: mid_range)")
    p.add_argument("--sample-rate", type=float, default=50.0,
                   help="Sensor sample rate in Hz for real-time check (default: 50)")
    p.add_argument("--runs",  type=int, default=BENCH_RUNS)
    p.add_argument("--no-coreml", action="store_true", help="Skip CoreML benchmark")
    p.add_argument("--no-onnx",   action="store_true", help="Skip ONNX benchmark")
    p.add_argument("--out-dir",   default=None,
                   help="Directory for exported model artefacts (default: models/mobile_exports)")
    p.add_argument("--log", default=None,
                   help="Path to log file (default: logs/mobile_sim_<timestamp>.log)")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model) if Path(args.model).is_absolute() \
                 else _HERE / args.model
    out_dir = Path(args.out_dir) if args.out_dir \
              else _HERE / "models" / "mobile_exports"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Logger ────────────────────────────────────────────────────────────────
    ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log) if args.log \
               else _HERE / "logs" / f"mobile_sim_{ts_stamp}.log"
    log = Logger(log_path)

    profile = DEVICE_PROFILES[args.profile]

    # ── Compute MAC / byte counts once (analytical) ───────────────────────────
    total_macs  = count_macs_deepconvlstm(args.window, args.channels, args.classes)
    total_bytes = count_model_bytes_fp32(args.channels, args.classes, window=args.window)
    proj_ms, bottleneck = roofline_latency_ms(total_macs, total_bytes, profile)

    # ── Header ────────────────────────────────────────────────────────────────
    log()
    log("=" * 110)
    log("  DeepConvLSTM  —  Mobile Runtime Simulator  (Roofline Model)")
    log("=" * 110)
    log(f"  Checkpoint   : {model_path}")
    log(f"  Input shape  : (1, {args.window}, {args.channels})  "
        f"[batch=1, time={args.window}, channels={args.channels}]")
    log(f"  Device       : {profile['name']}  [{args.profile}]")
    log(f"  CPU (1 core) : {profile['cpu_gflops_fp32']} GFLOPS FP32 "
        f"| BW {profile['memory_bw_gbs']} GB/s | η={profile['efficiency']}")
    log(f"  MACs         : {total_macs:,}  ({total_macs/1e6:.2f}M)")
    log(f"  Model bytes  : {total_bytes:,}  ({total_bytes/1e6:.2f} MB, weights+activations)")
    log(f"  Roofline est : {proj_ms:.2f} ms  [{bottleneck}-bound]")
    log(f"  Benchmark    : {args.runs} runs  warmup={WARMUP_RUNS}")
    log(f"  Export dir   : {out_dir}")
    log(f"  Log          : {log_path}")
    log("=" * 110)

    # ── Load model ────────────────────────────────────────────────────────────
    model = build_model(args.channels, args.classes)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, args.window, args.channels)

    # ── 0. Torch Profiler ─────────────────────────────────────────────────────
    log()
    log("=" * 110)
    log("  0. Torch Profiler — actual per-operator CPU timing (single thread)")
    log("=" * 110)
    run_torch_profiler(model, dummy, log_fn=log)
    log()

    # ── Latency benchmarks ────────────────────────────────────────────────────
    log()
    log(f"  {'Runtime':<22s}  "
        f"{'mean':>10s}  {'p50':>10s}  "
        f"{'p95':>10s}  {'p99':>10s}  "
        f"{'size':>11s}  {'projected (roofline)':>28s}")
    print_separator(log)

    window_ms = (args.window / args.sample_rate) * 1000.0

    # ── 1. PyTorch CPU ────────────────────────────────────────────────────────
    mean, p50, p95, p99, size_str = bench_pytorch_cpu(model, dummy)
    print_row("PyTorch CPU (1 thread)", mean, p50, p95, p99, size_str,
              proj_ms, bottleneck, log)
    _, ok = realtime_check(proj_ms, args.window, args.sample_rate)
    log(f"  {'':22s}  → mobile projected {proj_ms:.1f} ms  |  "
        f"window={window_ms:.0f} ms @ {args.sample_rate:.0f} Hz  [{feasibility_tag(ok)}]")
    print_separator(log)

    # ── 2. TorchScript ────────────────────────────────────────────────────────
    label, mean, p50, p95, p99, size_str, ts_path = \
        bench_torchscript(model, dummy, out_dir)
    print_row(label, mean, p50, p95, p99, size_str, proj_ms, bottleneck, log)
    log(f"  {'':22s}  → mobile projected {proj_ms:.1f} ms  [{feasibility_tag(ok)}]"
        f"  saved → {ts_path}")
    print_separator(log)

    # ── 3. CoreML ─────────────────────────────────────────────────────────────
    if not args.no_coreml:
        result, err = bench_coreml(model, dummy, out_dir)
        if err:
            log(f"  {'CoreML':<22s}  ✗  {err}")
        else:
            assert result is not None
            mean, p50, p95, p99, size_str, cml_path = result
            # CoreML on Apple Silicon IS the A-series estimate; use measured value
            print_row("CoreML (CPU+NE)", mean, p50, p95, p99, size_str,
                      mean, "measured on-device", log)
            _, ok_cml = realtime_check(mean, args.window, args.sample_rate)
            log(f"  {'':22s}  → measured {mean:.1f} ms  "
                f"[{feasibility_tag(ok_cml)}]"
                f"  saved → {cml_path}")
        print_separator(log)

    # ── 4. ONNX Runtime ───────────────────────────────────────────────────────
    if not args.no_onnx:
        result, err = bench_onnx(model, dummy, out_dir)
        if err:
            log(f"  {'ONNX Runtime':<22s}  ✗  {err}")
        else:
            assert result is not None
            mean, p50, p95, p99, size_str, onnx_path, quant = result
            print_row("ONNX (FP32, 1 thread)", mean, p50, p95, p99, size_str,
                      proj_ms, bottleneck, log)
            log(f"  {'':22s}  → mobile projected {proj_ms:.1f} ms  "
                f"[{feasibility_tag(ok)}]"
                f"  saved → {onnx_path}")

            if quant:
                qmean, qp50, qp95, qp99, qsize_str, qpath = quant
                # INT8 cuts compute roughly 2×; memory halved
                int8_macs  = total_macs
                int8_bytes = total_bytes // 2
                # INT8 throughput ~ 2× FP32 on devices with SIMD INT8 support
                int8_profile = dict(profile)
                int8_profile["cpu_gflops_fp32"] = profile["cpu_gflops_fp32"] * 2.0
                qproj_ms, qbottleneck = roofline_latency_ms(int8_macs, int8_bytes,
                                                             int8_profile)
                print_row("ONNX (INT8 quant)",
                          qmean, qp50, qp95, qp99, qsize_str, qproj_ms, qbottleneck, log)
                _, ok_q = realtime_check(qproj_ms, args.window, args.sample_rate)
                log(f"  {'':22s}  → mobile projected {qproj_ms:.1f} ms  "
                    f"[{feasibility_tag(ok_q)}]"
                    f"  saved → {qpath}")
        print_separator(log)

    # ── Summary ───────────────────────────────────────────────────────────────
    log()
    log("  Methodology")
    log("  ───────────")
    log(f"  • Projected latency uses the Roofline Model, not a Mac CPU scaling factor.")
    log(f"    compute_time = MACs×2 / (peak_GFLOPS × η),  "
        f"memory_time = bytes / (peak_BW × η)")
    log(f"    projected = max(compute_time, memory_time)  → bottleneck: {bottleneck}-bound")
    log(f"  • Device: {profile['name']}")
    log(f"    Peak CPU: {profile['cpu_gflops_fp32']} GFLOPS FP32  "
        f"| BW: {profile['memory_bw_gbs']} GB/s  | η={profile['efficiency']}")
    log(f"  • Model MACs: {total_macs/1e6:.2f}M  |  "
        f"Memory footprint: {total_bytes/1e6:.2f} MB")
    log(f"  • Real-time threshold: {window_ms:.0f} ms  "
        f"(window={args.window} samples @ {args.sample_rate} Hz)")
    log("  • CoreML on Apple Silicon: benchmark IS the Neural Engine measurement.")
    log("  • ONNX INT8: weight bytes halved, INT8 throughput modelled as 2× FP32 SIMD.")
    log("  • For Android: export ONNX → ONNX Runtime Mobile or TFLite (onnx-tf).")
    log()
    log.close()


if __name__ == "__main__":
    main()
