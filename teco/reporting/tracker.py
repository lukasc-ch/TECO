"""ProgressTracker: structured event collection for optimization runs.

The tracker is the single source of truth consumed by both the live console
reporter and the post-run HTML report generator.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


@dataclass
class PhaseEvent:
    """A high-level phase transition (INIT, PLANNING, ITERATION, FINALISE)."""

    phase: str
    timestamp: float  # time.perf_counter()
    message: str = ""


@dataclass
class BaselineEvent:
    """Baseline profiling result for a single shape point."""

    shape_label: str
    latency_ms: float
    tflops: float
    bottleneck: str | None = None  # from Stage 2, if available


@dataclass
class StrategyEvent:
    """Lifecycle event for a strategy."""

    strategy_id: str
    strategy_name: str
    action: str  # created, deepened, compiled, compile_failed, validated,
    #              correctness_failed, profiled, pruned, regime_winner
    iteration: int
    detail: str = ""


@dataclass
class IterationResult:
    """Complete result of a single optimization iteration."""

    iteration: int
    strategy_id: str
    strategy_name: str
    shape_results: dict[str, float]  # {shape_label: latency_ms}
    vs_baseline: dict[str, float]  # {shape_label: pct improvement (positive = faster)}
    compile_ok: bool
    correctness_ok: bool
    wall_time_s: float


@dataclass
class FinalSummary:
    """Summary emitted at the end of the run."""

    overall_speedup: float
    regime_winners: list[dict[str, str]]  # [{id, name, regime, tflops}]
    pruned_strategies: list[dict[str, str]]  # [{id, name, reason}]
    total_iterations: int
    wall_time_s: float
    report_path: str | None = None


# ---------------------------------------------------------------------------
# Listener protocol
# ---------------------------------------------------------------------------

# Listeners are callables that receive (event_type: str, event: Any).
Listener = Callable[[str, Any], None]


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------


@dataclass
class ProgressTracker:
    """Collects structured events throughout an optimization run.

    Attach listeners (e.g. RichConsoleReporter) via ``add_listener()``.
    Each ``emit_*`` method appends to the internal log and notifies listeners.
    """

    run_id: str = ""
    start_time: float = field(default_factory=time.perf_counter)
    max_iterations: int = 10

    # Event stores
    phase_events: list[PhaseEvent] = field(default_factory=list)
    baseline_events: list[BaselineEvent] = field(default_factory=list)
    strategy_events: list[StrategyEvent] = field(default_factory=list)
    iteration_results: list[IterationResult] = field(default_factory=list)
    final_summary: FinalSummary | None = None

    # Hardware info (set once during INIT)
    device_name: str = ""
    peak_tflops_fp16: float = 0.0
    peak_bandwidth_gbs: float = 0.0

    # Listeners
    _listeners: list[Listener] = field(default_factory=list, repr=False)

    def add_listener(self, listener: Listener) -> None:
        self._listeners.append(listener)

    def _notify(self, event_type: str, event: Any) -> None:
        for listener in self._listeners:
            with contextlib.suppress(Exception):
                listener(event_type, event)

    # ── Emit methods ──────────────────────────────────────────────────────

    def emit_phase(self, phase: str, message: str = "") -> None:
        ev = PhaseEvent(phase=phase, timestamp=time.perf_counter(), message=message)
        self.phase_events.append(ev)
        self._notify("phase", ev)

    def emit_baseline(
        self,
        shape_label: str,
        latency_ms: float,
        tflops: float,
        bottleneck: str | None = None,
    ) -> None:
        ev = BaselineEvent(
            shape_label=shape_label,
            latency_ms=latency_ms,
            tflops=tflops,
            bottleneck=bottleneck,
        )
        self.baseline_events.append(ev)
        self._notify("baseline", ev)

    def emit_strategy(
        self,
        strategy_id: str,
        strategy_name: str,
        action: str,
        iteration: int,
        detail: str = "",
    ) -> None:
        ev = StrategyEvent(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            action=action,
            iteration=iteration,
            detail=detail,
        )
        self.strategy_events.append(ev)
        self._notify("strategy", ev)

    def record_iteration(
        self,
        iteration: int,
        strategy_id: str,
        strategy_name: str,
        shape_results: dict[str, float],
        vs_baseline: dict[str, float],
        compile_ok: bool,
        correctness_ok: bool,
        wall_time_s: float,
    ) -> None:
        result = IterationResult(
            iteration=iteration,
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            shape_results=shape_results,
            vs_baseline=vs_baseline,
            compile_ok=compile_ok,
            correctness_ok=correctness_ok,
            wall_time_s=wall_time_s,
        )
        self.iteration_results.append(result)
        self._notify("iteration_result", result)

    def emit_final(
        self,
        overall_speedup: float,
        regime_winners: list[dict[str, str]],
        pruned_strategies: list[dict[str, str]],
        total_iterations: int,
        report_path: str | None = None,
    ) -> None:
        self.final_summary = FinalSummary(
            overall_speedup=overall_speedup,
            regime_winners=regime_winners,
            pruned_strategies=pruned_strategies,
            total_iterations=total_iterations,
            wall_time_s=self.elapsed(),
            report_path=report_path,
        )
        self._notify("final", self.final_summary)

    # ── Queries ───────────────────────────────────────────────────────────

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot of all tracked data (for HTML report)."""
        return {
            "run_id": self.run_id,
            "device_name": self.device_name,
            "peak_tflops_fp16": self.peak_tflops_fp16,
            "peak_bandwidth_gbs": self.peak_bandwidth_gbs,
            "max_iterations": self.max_iterations,
            "wall_time_s": self.elapsed(),
            "baseline": [
                {
                    "shape": e.shape_label,
                    "latency_ms": e.latency_ms,
                    "tflops": e.tflops,
                    "bottleneck": e.bottleneck,
                }
                for e in self.baseline_events
            ],
            "strategies_created": [
                {
                    "id": e.strategy_id,
                    "name": e.strategy_name,
                    "detail": e.detail,
                }
                for e in self.strategy_events
                if e.action == "created"
            ],
            "iterations": [
                {
                    "iteration": r.iteration,
                    "strategy_id": r.strategy_id,
                    "strategy_name": r.strategy_name,
                    "shape_results": r.shape_results,
                    "vs_baseline": r.vs_baseline,
                    "compile_ok": r.compile_ok,
                    "correctness_ok": r.correctness_ok,
                    "wall_time_s": r.wall_time_s,
                }
                for r in self.iteration_results
            ],
            "final": (
                {
                    "overall_speedup": self.final_summary.overall_speedup,
                    "regime_winners": self.final_summary.regime_winners,
                    "pruned_strategies": self.final_summary.pruned_strategies,
                    "total_iterations": self.final_summary.total_iterations,
                    "wall_time_s": self.final_summary.wall_time_s,
                    "report_path": self.final_summary.report_path,
                }
                if self.final_summary
                else None
            ),
        }
