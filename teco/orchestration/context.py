"""OptimizationContext: shared state passed between agents and the orchestration loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from teco.knowledge.schema import (
    KernelCharacteristics,
    ProfilingReport,
    QueryResult,
    ShapePoint,
    Strategy,
    StrategyTree,
)
from teco.reporting.tracker import ProgressTracker
from teco.tools.profiler import HardwareCeilings


@dataclass
class OptimizationContext:
    # ── Task description ────────────────────────────────────────────────────
    task_name: str
    language: str  # "triton" | "cuda" | "tilelang" | "ascendc"
    kernel_path: Path
    kernel_source: str  # current (possibly modified) source

    # ── Hardware ─────────────────────────────────────────────────────────────
    ceilings: HardwareCeilings

    # ── Shape sweep ──────────────────────────────────────────────────────────
    shape_sweep: list[ShapePoint] = field(default_factory=list)

    # ── Profiling ────────────────────────────────────────────────────────────
    # Stage 1 latency per shape: {shape_label: {tflops, latency_ms, ...}}
    baseline_stage1: dict[str, dict[str, float]] = field(default_factory=dict)
    # Stage 2 deep ncu for baseline: {shape_label: ProfilingReport}
    baseline_stage2: dict[str, ProfilingReport] = field(default_factory=dict)

    # ── Knowledge query results ───────────────────────────────────────────────
    knowledge_query: QueryResult | None = None

    # ── Kernel characteristics (extracted by agent) ───────────────────────────
    kernel_characteristics: KernelCharacteristics | None = None

    # ── Strategy tree ─────────────────────────────────────────────────────────
    strategy_tree: StrategyTree = field(default_factory=StrategyTree)

    # ── Iteration state ───────────────────────────────────────────────────────
    iteration: int = 0
    max_iterations: int = 10
    plateau_threshold: float = 0.02  # stop if improvement < 2% for 2 consecutive iters
    target_efficiency_pct: float = 90.0  # stop when within this % of hardware ceiling

    # ── Run record state ──────────────────────────────────────────────────────
    run_id: str = ""
    ncu_output_dir: Path = field(default_factory=lambda: Path("knowledge/runs/ncu"))

    # ── Progress tracking ─────────────────────────────────────────────────────
    tracker: ProgressTracker = field(default_factory=ProgressTracker)

    # ── Misc ─────────────────────────────────────────────────────────────────
    verbosity: int = 1  # 0=quiet  1=normal  2=verbose  3=debug
    generate_report: bool = True  # whether to produce an HTML report after the run
    extra: dict[str, Any] = field(default_factory=dict)

    def log(self, msg: str, level: int = 1) -> None:
        """Print msg if verbosity >= level."""
        if self.verbosity >= level:
            print(msg)

    def best_latency_across_shapes(self) -> float:
        """Return the best (lowest) latency_ms achieved across all shapes and strategies.

        Returns float('inf') if no results exist.
        """
        best = float("inf")
        for strategy in self.strategy_tree.strategies:
            for result in strategy.shape_results:
                if result.latency_ms > 0:
                    best = min(best, result.latency_ms)
        return best

    def baseline_latency(self, shape_label: str = "medium") -> float:
        """Return baseline latency_ms for a given shape label."""
        return self.baseline_stage1.get(shape_label, {}).get("latency_ms", 0.0)

    def baseline_tflops(self, shape_label: str = "medium") -> float:
        """Return baseline TFLOPS for a given shape label (for roofline reporting only)."""
        return self.baseline_stage1.get(shape_label, {}).get("tflops", 0.0)

    def primary_bottleneck(self) -> str:
        """Return the bottleneck reported at the 'medium' shape in baseline Stage 2."""
        report = self.baseline_stage2.get("medium")
        if report:
            return report.bottleneck
        return "unknown"

    def latency_speedup(self, new_latency: float, shape_label: str = "medium") -> float:
        """Compute speedup as baseline_latency / new_latency (higher is better)."""
        baseline = self.baseline_latency(shape_label)
        if baseline <= 0 or new_latency <= 0:
            return 1.0
        return baseline / new_latency

    def efficiency_pct(self, tflops: float, dtype: str = "fp16") -> float:
        """Compute efficiency as % of hardware peak TFLOPS (for roofline reporting only)."""
        peak = self.ceilings.peak_tflops(dtype)
        return (tflops / peak * 100) if peak > 0 else 0.0
