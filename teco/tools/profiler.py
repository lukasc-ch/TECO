"""Two-stage GPU profiling tool for TECO.

Stage 1: triton.testing.do_bench — fast latency measurement, every iteration.
Stage 2: ncu (Nsight Compute) — deep metric extraction, baseline + final only.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Callable

import pynvml
import torch

from teco.knowledge.schema import ProfilingReport, ShapePoint, StrategyShapeResult


# ---------------------------------------------------------------------------
# Hardware info (queried once at startup)
# ---------------------------------------------------------------------------


class HardwareCeilings:
    """GPU hardware limits, queried via pynvml."""

    def __init__(self, device_index: int = 0) -> None:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        self.device_name: str = pynvml.nvmlDeviceGetName(handle)
        # Memory bandwidth (theoretical, from specs)
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        self.total_memory_gb: float = mem_info.total / 1e9
        # Peak bandwidth and TFLOPS are hardware-specific; read from ncu or hardcode fallback
        self.peak_bandwidth_gbs: float = _lookup_peak_bandwidth(self.device_name)
        self.peak_tflops_fp16: float = _lookup_peak_tflops(self.device_name, "fp16")
        self.peak_tflops_fp32: float = _lookup_peak_tflops(self.device_name, "fp32")
        pynvml.nvmlShutdown()

    def peak_tflops(self, dtype: str) -> float:
        if dtype in ("fp16", "bf16"):
            return self.peak_tflops_fp16
        return self.peak_tflops_fp32


# ---------------------------------------------------------------------------
# Stage 1: fast latency profiling
# ---------------------------------------------------------------------------


def profile_latency(
    fn: Callable[[], Any],
    flops: int | None = None,
    bytes_transferred: int | None = None,
    warmup: int = 25,
    rep: int = 100,
) -> dict[str, float]:
    """Run triton.testing.do_bench and return basic metrics.

    Args:
        fn: Zero-argument callable that runs the kernel.
        flops: Total FLOPs for one kernel invocation (for TFLOPS calculation).
        bytes_transferred: Total bytes read+written (for GB/s calculation).
        warmup: Warm-up iterations.
        rep: Measurement repetitions.

    Returns:
        Dict with latency_ms, tflops (if flops given), bandwidth_gbs (if bytes given).
    """
    try:
        from triton.testing import do_bench
    except ImportError as e:
        raise RuntimeError("triton must be installed for Stage 1 profiling") from e

    latency_ms: float = do_bench(fn, warmup=warmup, rep=rep)
    result: dict[str, float] = {"latency_ms": latency_ms}

    if flops is not None and latency_ms > 0:
        result["tflops"] = (flops * 1e-12) / (latency_ms * 1e-3)
    if bytes_transferred is not None and latency_ms > 0:
        result["bandwidth_gbs"] = (bytes_transferred * 1e-9) / (latency_ms * 1e-3)

    return result


def profile_shape_sweep_stage1(
    make_fn: Callable[[ShapePoint], tuple[Callable[[], Any], int, int]],
    shape_sweep: list[ShapePoint],
    baseline_results: dict[str, dict[str, float]] | None = None,
    warmup: int = 25,
    rep: int = 100,
) -> list[StrategyShapeResult]:
    """
    Run Stage 1 profiling across all shapes in the sweep.

    Args:
        make_fn: Given a ShapePoint, returns (fn, flops, bytes_transferred).
        shape_sweep: List of shape points to profile.
        baseline_results: If provided, compute vs_baseline_pct per shape.
        warmup/rep: Passed to do_bench.

    Returns:
        List of StrategyShapeResult (one per shape point).
    """
    results: list[StrategyShapeResult] = []
    for shape in shape_sweep:
        fn, flops, nbytes = make_fn(shape)
        metrics = profile_latency(fn, flops=flops, bytes_transferred=nbytes, warmup=warmup, rep=rep)
        tflops = metrics.get("tflops", 0.0)

        baseline_tflops = 0.0
        if baseline_results and shape.label in baseline_results:
            baseline_tflops = baseline_results[shape.label].get("tflops", 0.0)
        vs_baseline = (
            ((tflops - baseline_tflops) / baseline_tflops * 100) if baseline_tflops > 0 else 0.0
        )

        results.append(
            StrategyShapeResult(
                shape_point=shape,
                achieved_tflops=tflops,
                vs_baseline_pct=vs_baseline,
                profiling_report=None,  # filled by Stage 2
            )
        )
    return results


# ---------------------------------------------------------------------------
# Stage 2: deep ncu profiling
# ---------------------------------------------------------------------------

# ncu metrics we request. These metric names are for Nsight Compute >= 2022.
_NCU_METRICS = [
    # Compute throughput
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    # FLOP counts (fp16/fp32)
    "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
    # Tensor cores
    "sm__inst_executed_pipe_tensor.sum",
    # Memory hierarchy
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
    "l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum",
    # Shared memory bank conflicts
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    # Warp efficiency
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "smsp__sass_average_branch_targets_threads_uniform.pct",
    # Occupancy
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__maximum_warps_per_active_cycle_pct",
    # Instruction replay
    "smsp__inst_executed.sum",
    "smsp__inst_issued.sum",
]

_NCU_METRICS_STR = ",".join(_NCU_METRICS)


def profile_deep_ncu(
    runner_script: str,
    output_dir: Path,
    label: str,
    ceilings: HardwareCeilings,
    extra_args: list[str] | None = None,
) -> ProfilingReport | None:
    """
    Run ncu on a Python runner script and parse the CSV output.

    Args:
        runner_script: Python source code that imports and runs the kernel once.
        output_dir: Directory to write ncu report files.
        label: Short label for this profiling run (used in filenames).
        ceilings: Hardware ceilings from pynvml.
        extra_args: Additional ncu CLI arguments.

    Returns:
        Parsed ProfilingReport or None if ncu is not available / failed.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{label}.ncu-rep"
    csv_path = output_dir / f"{label}.csv"

    # Write runner script to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, prefix="teco_ncu_runner_"
    ) as f:
        f.write(runner_script)
        runner_path = f.name

    cmd = [
        "ncu",
        "--set", "full",
        "--metrics", _NCU_METRICS_STR,
        "--csv",
        "--log-file", str(csv_path),
        "--export", str(report_path),
        "--target-processes", "all",
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["python", runner_path])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except FileNotFoundError:
        return None  # ncu not on PATH
    except subprocess.TimeoutExpired:
        return None
    finally:
        Path(runner_path).unlink(missing_ok=True)

    if result.returncode != 0 and not csv_path.exists():
        return None

    return _parse_ncu_csv(csv_path, ceilings, str(report_path))


def _parse_ncu_csv(
    csv_path: Path, ceilings: HardwareCeilings, report_path: str
) -> ProfilingReport | None:
    """Parse ncu CSV output into a ProfilingReport."""
    if not csv_path.exists():
        return None

    # ncu CSV has multiple header lines; rows are kernel invocations
    # Format: "ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",...,metric_name,...
    text = csv_path.read_text(errors="replace")
    # Strip ncu preamble lines (lines not starting with a quote or digit)
    lines = [l for l in text.splitlines() if l.startswith('"') or (l and l[0].isdigit())]
    if not lines:
        return None

    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = list(reader)
    if not rows:
        return None

    # Aggregate across kernel invocations (take mean of numeric metrics)
    agg: dict[str, list[float]] = {}
    for row in rows:
        for k, v in row.items():
            try:
                agg.setdefault(k, []).append(float(v.replace(",", "")))
            except (ValueError, AttributeError):
                pass

    def m(key: str) -> float:
        vals = agg.get(key, [])
        return sum(vals) / len(vals) if vals else 0.0

    # Derived metrics
    fp32_ops = int(
        m("sm__sass_thread_inst_executed_op_fadd_pred_on.sum")
        + m("sm__sass_thread_inst_executed_op_fmul_pred_on.sum")
        + 2 * m("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
    )
    fp16_ops = int(
        m("sm__sass_thread_inst_executed_op_hadd_pred_on.sum")
        + m("sm__sass_thread_inst_executed_op_hmul_pred_on.sum")
        + 2 * m("sm__sass_thread_inst_executed_op_hfma_pred_on.sum")
    )
    dram_bytes = m("dram__bytes_read.sum") + m("dram__bytes_write.sum")
    bank_conflicts = int(
        m("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum")
        + m("l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum")
    )

    compute_pct = m("sm__throughput.avg.pct_of_peak_sustained_elapsed")
    memory_pct = m("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
    if compute_pct > memory_pct and compute_pct > 50:
        bottleneck = "compute"
    elif memory_pct > 50:
        bottleneck = "memory"
    else:
        bottleneck = "latency"

    replay_ratio = (
        m("smsp__inst_issued.sum") / m("smsp__inst_executed.sum")
        if m("smsp__inst_executed.sum") > 0
        else 1.0
    )

    return ProfilingReport(
        achieved_tflops=0.0,  # filled externally from Stage 1
        peak_tflops=ceilings.peak_tflops_fp16,
        arithmetic_intensity=0.0,  # filled externally (FLOP / bytes)
        bottleneck=bottleneck,  # type: ignore[arg-type]
        utilization_pct=max(compute_pct, memory_pct),
        l1_hit_rate=m("l1tex__t_sector_hit_rate.pct") / 100,
        l2_hit_rate=m("lts__t_sector_hit_rate.pct") / 100,
        dram_bandwidth_gbs=0.0,  # derived from dram_bytes / latency; filled externally
        peak_bandwidth_gbs=ceilings.peak_bandwidth_gbs,
        shared_mem_bank_conflicts=bank_conflicts,
        global_load_efficiency=_safe_ratio(
            m("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum"), dram_bytes
        ),
        global_store_efficiency=_safe_ratio(
            m("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum"), dram_bytes
        ),
        warp_efficiency=m("smsp__thread_inst_executed_per_inst_executed.ratio") / 100,
        branch_divergence_pct=100.0
        - m("smsp__sass_average_branch_targets_threads_uniform.pct"),
        achieved_occupancy=m("sm__warps_active.avg.pct_of_peak_sustained_active") / 100,
        theoretical_occupancy=m("sm__maximum_warps_per_active_cycle_pct") / 100,
        instruction_replay_overhead=max(0.0, replay_ratio - 1.0),
        tensor_core_utilization=compute_pct / 100 if bottleneck == "compute" else 0.0,
        fp16_ops=fp16_ops,
        fp32_ops=fp32_ops,
        int_ops=0,
        special_func_ops=0,
        latency_ms=0.0,  # filled from Stage 1
        raw_ncu_report_path=report_path,
    )


def merge_stage1_into_report(
    report: ProfilingReport,
    stage1: dict[str, float],
    flops: int,
    bytes_transferred: int,
) -> ProfilingReport:
    """Fill achieved_tflops, latency_ms, arithmetic_intensity, dram_bandwidth_gbs
    from Stage 1 results into a Stage 2 ProfilingReport."""
    latency_ms = stage1.get("latency_ms", 0.0)
    tflops = stage1.get("tflops", 0.0)
    bw_gbs = stage1.get("bandwidth_gbs", 0.0)
    ai = (flops / bytes_transferred) if bytes_transferred > 0 else 0.0
    return report.model_copy(
        update={
            "achieved_tflops": tflops,
            "arithmetic_intensity": ai,
            "dram_bandwidth_gbs": bw_gbs,
            "latency_ms": latency_ms,
        }
    )


# ---------------------------------------------------------------------------
# Hardware spec lookup (fallback table for common GPUs)
# ---------------------------------------------------------------------------

_BANDWIDTH_TABLE: dict[str, float] = {
    "a100": 2039.0,
    "h100": 3350.0,
    "a10": 600.0,
    "a30": 933.0,
    "v100": 900.0,
    "rtx 4090": 1008.0,
    "rtx 3090": 936.0,
}

_TFLOPS_TABLE: dict[str, dict[str, float]] = {
    "a100": {"fp16": 312.0, "fp32": 19.5},
    "h100": {"fp16": 989.0, "fp32": 67.0},
    "a10": {"fp16": 125.0, "fp32": 31.2},
    "a30": {"fp16": 165.0, "fp32": 10.3},
    "v100": {"fp16": 112.0, "fp32": 14.0},
    "rtx 4090": {"fp16": 165.0, "fp32": 82.6},
    "rtx 3090": {"fp16": 71.0, "fp32": 35.6},
}


def _lookup_peak_bandwidth(device_name: str) -> float:
    name_lower = device_name.lower()
    for key, bw in _BANDWIDTH_TABLE.items():
        if key in name_lower:
            return bw
    return 900.0  # conservative fallback


def _lookup_peak_tflops(device_name: str, dtype: str) -> float:
    name_lower = device_name.lower()
    for key, table in _TFLOPS_TABLE.items():
        if key in name_lower:
            return table.get(dtype, 10.0)
    return 10.0  # conservative fallback


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0
