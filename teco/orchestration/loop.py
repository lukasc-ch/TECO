"""Main optimization loop: coordinates agents and builds OptimizationContext."""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from pathlib import Path

from teco.agents.experimenter import ExperimenterAgent
from teco.agents.learner import LearnerAgent
from teco.agents.optimizer import OptimizerAgent, _make_hardware_id
from teco.agents.reflector import ReflectorAgent
from teco.knowledge.schema import ShapePoint
from teco.knowledge.store import KnowledgeStore
from teco.orchestration.context import OptimizationContext
from teco.reporting.console import RichConsoleReporter
from teco.reporting.html_report import HTMLReportGenerator
from teco.reporting.tracker import ProgressTracker
from teco.tools.code_editor import read_kernel_file
from teco.tools.profiler import HardwareCeilings


def run_optimization(
    kernel_path: Path,
    *,
    language: str = "triton",
    model_id: str = "claude-opus-4-6",
    knowledge_root: Path = Path("knowledge"),
    max_iterations: int = 10,
    target_efficiency_pct: float = 90.0,
    shape_sweep: list[ShapePoint] | None = None,
    verbosity: int = 1,
    generate_report: bool = True,
) -> OptimizationContext:
    """
    Top-level entry point for a full optimization run.

    Args:
        kernel_path: Path to the Triton (or other language) kernel file.
        language: Source language identifier.
        model_id: Anthropic model to use for all agents.
        knowledge_root: Root of the knowledge store directory.
        max_iterations: Max optimization iterations.
        target_efficiency_pct: Stop when this % of hardware ceiling is reached.
        shape_sweep: Override the default small/medium/large shape sweep.
        verbosity: Output verbosity level (0=quiet, 1=normal, 2=verbose, 3=debug).
        generate_report: Whether to produce an HTML report after the run.

    Returns:
        The final OptimizationContext containing all results.
    """
    kernel_source = read_kernel_file(kernel_path)
    task_name = _slugify(kernel_path.stem)
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{task_name}"

    # Hardware
    ceilings = HardwareCeilings()

    # Default shape sweep
    if shape_sweep is None:
        shape_sweep = _default_shape_sweep(kernel_source)

    # Knowledge store
    store = KnowledgeStore(knowledge_root=knowledge_root)

    # Progress tracker + console reporter
    tracker = ProgressTracker(
        run_id=run_id,
        max_iterations=max_iterations,
        device_name=ceilings.device_name,
        peak_tflops_fp16=ceilings.peak_tflops_fp16,
        peak_bandwidth_gbs=ceilings.peak_bandwidth_gbs,
    )
    reporter = RichConsoleReporter(tracker, verbosity=verbosity)
    tracker.add_listener(reporter)

    # Build context
    context = OptimizationContext(
        task_name=task_name,
        language=language,
        kernel_path=kernel_path,
        kernel_source=kernel_source,
        ceilings=ceilings,
        shape_sweep=shape_sweep,
        max_iterations=max_iterations,
        target_efficiency_pct=target_efficiency_pct,
        run_id=run_id,
        ncu_output_dir=knowledge_root / "runs" / "ncu",
        tracker=tracker,
        verbosity=verbosity,
        generate_report=generate_report,
    )

    # ── Agent pipeline ─────────────────────────────────────────────────────

    # Phase 1: LearnerAgent (stub — checks if new hardware/language needs doc research)
    learner = LearnerAgent(model_id=model_id)
    context = learner.run(context)

    # Phase 2: OptimizerAgent (full MVP)
    optimizer = OptimizerAgent(store=store, model_id=model_id)
    context = optimizer.run(context)

    # Phase 3: ExperimenterAgent (stub — invoked on ambiguous bottlenecks in Phase 2)
    experimenter = ExperimenterAgent(model_id=model_id)
    context = experimenter.run(context)

    # Phase 4: ReflectorAgent (stub — lesson synthesis)
    reflector = ReflectorAgent(model_id=model_id)
    context = reflector.run(context)

    # ── HTML report generation ─────────────────────────────────────────────
    if generate_report:
        report_gen = HTMLReportGenerator()
        report_path = report_gen.generate(tracker, output_dir=knowledge_root / "runs")
        # Update the final summary with the report path
        if tracker.final_summary:
            tracker.final_summary.report_path = str(report_path)

    return context


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_shape_sweep(kernel_source: str) -> list[ShapePoint]:
    """
    Attempt to infer shape parameters from the kernel source and return a
    small/medium/large sweep. Falls back to generic M/N/K sweep if not parseable.
    """
    # Look for common dimension patterns in triton kernels
    params: list[str] = re.findall(
        r"\bM\b|\bN\b|\bK\b|\bseq_len\b|\bhead_dim\b|\bbatch_size\b|\bseq_q\b|\bseq_k\b",
        kernel_source,
    )
    unique_params = list(dict.fromkeys(params))  # preserve order, deduplicate

    if "seq_q" in unique_params or "seq_len" in unique_params:
        # Attention-like kernel
        seq_key = "seq_q" if "seq_q" in unique_params else "seq_len"
        return [
            ShapePoint(label="small", dims={"batch": 1, "heads": 8, seq_key: 64, "head_dim": 64}),
            ShapePoint(label="medium", dims={"batch": 4, "heads": 16, seq_key: 512, "head_dim": 64}),
            ShapePoint(label="large", dims={"batch": 8, "heads": 32, seq_key: 4096, "head_dim": 128}),
        ]
    elif "K" in unique_params or ("M" in unique_params and "N" in unique_params):
        # GEMM-like kernel
        return [
            ShapePoint(label="small", dims={"M": 64, "N": 64, "K": 64}),
            ShapePoint(label="medium", dims={"M": 512, "N": 512, "K": 512}),
            ShapePoint(label="large", dims={"M": 4096, "N": 4096, "K": 1024}),
        ]
    else:
        # Generic fallback
        return [
            ShapePoint(label="small", dims={"N": 1024}),
            ShapePoint(label="medium", dims={"N": 65536}),
            ShapePoint(label="large", dims={"N": 4194304}),
        ]


def _slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
