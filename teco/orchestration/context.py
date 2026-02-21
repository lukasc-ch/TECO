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

    # ── Misc ─────────────────────────────────────────────────────────────────
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def best_tflops_across_shapes(self) -> float:
        """Return the best TFLOPS achieved in Stage 1 across all shapes and strategies."""
        best = 0.0
        for strategy in self.strategy_tree.strategies:
            for result in strategy.shape_results:
                best = max(best, result.achieved_tflops)
        return best

    def baseline_tflops(self, shape_label: str = "medium") -> float:
        return self.baseline_stage1.get(shape_label, {}).get("tflops", 0.0)

    def primary_bottleneck(self) -> str:
        """Return the bottleneck reported at the 'medium' shape in baseline Stage 2."""
        report = self.baseline_stage2.get("medium")
        if report:
            return report.bottleneck
        return "unknown"

    def efficiency_pct(self, tflops: float, dtype: str = "fp16") -> float:
        peak = self.ceilings.peak_tflops(dtype)
        return (tflops / peak * 100) if peak > 0 else 0.0
