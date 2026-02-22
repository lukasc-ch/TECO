"""RichConsoleReporter: live terminal output using the ``rich`` library.

Falls back to plain ``print()`` if ``rich`` is not installed.
"""

from __future__ import annotations

from typing import Any

from teco.reporting.tracker import (
    BaselineEvent,
    FinalSummary,
    IterationResult,
    PhaseEvent,
    ProgressTracker,
    StrategyEvent,
)

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, TextColumn, TimeElapsedColumn
    from rich.table import Table

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


class RichConsoleReporter:
    """Observer that renders ProgressTracker events to the terminal.

    Instantiate and register as a listener on the tracker::

        reporter = RichConsoleReporter(tracker, verbosity=1)
        tracker.add_listener(reporter)
    """

    def __init__(self, tracker: ProgressTracker, verbosity: int = 1) -> None:
        self.tracker = tracker
        self.verbosity = verbosity
        self._console = Console() if HAS_RICH else None  # type: ignore[assignment]
        self._progress: Progress | None = None
        self._progress_task: Any = None
        self._baseline_rows: list[BaselineEvent] = []

    # ── Listener entry point ──────────────────────────────────────────────

    def __call__(self, event_type: str, event: Any) -> None:
        handler = {
            "phase": self._on_phase,
            "baseline": self._on_baseline,
            "strategy": self._on_strategy,
            "iteration_result": self._on_iteration_result,
            "final": self._on_final,
        }.get(event_type)
        if handler:
            handler(event)

    # ── Phase transitions ─────────────────────────────────────────────────

    def _on_phase(self, event: PhaseEvent) -> None:
        if event.phase == "INIT":
            self._print_header()
        self._print_phase_banner(event.phase, event.message)

        # Start progress bar for iteration loop
        if event.phase == "ITERATION" and HAS_RICH and self._console:
            self._progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self._console,
            )
            self._progress.start()
            self._progress_task = self._progress.add_task(
                "Iterations", total=self.tracker.max_iterations
            )

    # ── Baseline profiling ────────────────────────────────────────────────

    def _on_baseline(self, event: BaselineEvent) -> None:
        self._baseline_rows.append(event)

    def flush_baseline(self) -> None:
        """Render the baseline table. Called after all baseline events are emitted."""
        if not self._baseline_rows:
            return

        if HAS_RICH and self._console:
            table = Table(title="Baseline Performance", show_header=True, header_style="bold")
            table.add_column("Shape", style="cyan", width=12)
            table.add_column("Latency (ms)", justify="right", width=14)
            table.add_column("TFLOPS", justify="right", width=10)
            table.add_column("Bottleneck", width=14)
            for row in self._baseline_rows:
                bn = row.bottleneck or ""
                table.add_row(
                    row.shape_label,
                    f"{row.latency_ms:.2f}",
                    f"{row.tflops:.1f}",
                    bn,
                )
            self._console.print(table)
        else:
            self._print(f"\n{'Shape':<12}  {'Latency (ms)':>12}  {'TFLOPS':>8}  {'Bottleneck':>12}")
            self._print("─" * 50)
            for row in self._baseline_rows:
                bn = row.bottleneck or ""
                self._print(f"{row.shape_label:<12}  {row.latency_ms:>12.2f}  {row.tflops:>8.1f}  {bn:>12}")

    # ── Strategy events ───────────────────────────────────────────────────

    def _on_strategy(self, event: StrategyEvent) -> None:
        if event.action == "created" and self.verbosity >= 1:
            self._print(
                f"  {event.strategy_id}  {event.strategy_name:<28}  {event.detail}",
                style="dim" if not HAS_RICH else None,
            )
        elif event.action == "pruned" and self.verbosity >= 1:
            self._print(
                f"  ✗ [{event.strategy_id}] {event.strategy_name} — PRUNED: {event.detail}",
                style="dim red",
            )
        elif event.action == "regime_winner" and self.verbosity >= 1:
            self._print(
                f"  ★ [{event.strategy_id}] {event.strategy_name} — REGIME WINNER: {event.detail}",
                style="bold green",
            )
        elif event.action == "compile_failed" and self.verbosity >= 1:
            self._print(
                f"  ✗ [{event.strategy_id}] compile failed: {event.detail[:120]}",
                style="bold red",
            )
        elif event.action == "correctness_failed" and self.verbosity >= 1:
            self._print(
                f"  ✗ [{event.strategy_id}] correctness failed: {event.detail[:120]}",
                style="bold red",
            )

    # ── Iteration results ─────────────────────────────────────────────────

    def _on_iteration_result(self, result: IterationResult) -> None:
        # Advance progress bar
        if self._progress and self._progress_task is not None:
            self._progress.update(self._progress_task, advance=1)

        if self.verbosity < 1:
            return

        compile_icon = "✓" if result.compile_ok else "✗"
        correct_icon = "✓" if result.correctness_ok else "✗"

        # Build per-shape summary
        shape_parts: list[str] = []
        for label, tflops in result.shape_results.items():
            pct = result.vs_baseline.get(label, 0.0)
            if HAS_RICH:
                color = "green" if pct > 0 else ("red" if pct < 0 else "white")
                shape_parts.append(f"[{color}]{label}: {tflops:.1f} ({pct:+.1f}%)[/{color}]")
            else:
                shape_parts.append(f"{label}: {tflops:.1f} ({pct:+.1f}%)")

        line = (
            f"  [{result.strategy_id}] {result.strategy_name} → "
            f"{compile_icon} compile {correct_icon} correct  "
            + "  ".join(shape_parts)
        )

        if HAS_RICH and self._console:
            # Temporarily pause progress bar to print cleanly
            if self._progress:
                self._progress.stop()
            self._console.print(line)
            if self._progress:
                self._progress.start()
        else:
            print(line)

        # Verbose: per-shape table
        if self.verbosity >= 2 and HAS_RICH and self._console:
            table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
            table.add_column("Shape", style="cyan", width=10)
            table.add_column("TFLOPS", justify="right", width=10)
            table.add_column("vs baseline", justify="right", width=12)
            for label, tflops in result.shape_results.items():
                pct = result.vs_baseline.get(label, 0.0)
                color = "green" if pct > 0 else ("red" if pct < 0 else "white")
                table.add_row(label, f"{tflops:.1f}", f"[{color}]{pct:+.1f}%[/{color}]")
            if self._progress:
                self._progress.stop()
            self._console.print(table)
            if self._progress:
                self._progress.start()

    # ── Final summary ─────────────────────────────────────────────────────

    def _on_final(self, summary: FinalSummary) -> None:
        # Stop progress bar
        if self._progress:
            self._progress.stop()
            self._progress = None

        if HAS_RICH and self._console:
            lines: list[str] = [
                f"[bold]Run ID:[/bold]     {self.tracker.run_id}",
                f"[bold]Duration:[/bold]   {self._fmt_duration(summary.wall_time_s)}",
                f"[bold]Iterations:[/bold] {summary.total_iterations}/{self.tracker.max_iterations}",
                "",
                f"[bold green]Overall speedup: {summary.overall_speedup:.2f}x[/bold green]",
            ]
            if summary.regime_winners:
                lines.append("")
                lines.append("[bold]Regime winners:[/bold]")
                for w in summary.regime_winners:
                    lines.append(
                        f"  [green]★[/green] {w.get('id', '?')} {w.get('name', '?')} "
                        f"→ {w.get('regime', '?')} ({w.get('tflops', '?')} TFLOPS)"
                    )
            if summary.pruned_strategies:
                lines.append("")
                lines.append("[bold]Pruned:[/bold]")
                for p in summary.pruned_strategies:
                    lines.append(
                        f"  [dim]{p.get('id', '?')} {p.get('name', '?')} — {p.get('reason', '?')}[/dim]"
                    )
            if summary.report_path:
                lines.append("")
                lines.append(f"[bold]Report:[/bold] {summary.report_path}")

            panel = Panel(
                "\n".join(lines),
                title="[bold cyan]Run Summary[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
            self._console.print(panel)
        else:
            print("\n" + "=" * 60)
            print("  RUN SUMMARY")
            print("=" * 60)
            print(f"  Run ID:     {self.tracker.run_id}")
            print(f"  Duration:   {self._fmt_duration(summary.wall_time_s)}")
            print(f"  Iterations: {summary.total_iterations}/{self.tracker.max_iterations}")
            print(f"  Overall speedup: {summary.overall_speedup:.2f}x")
            if summary.regime_winners:
                print("  Regime winners:")
                for w in summary.regime_winners:
                    print(f"    ★ {w.get('id')} {w.get('name')} → {w.get('regime')} ({w.get('tflops')} TFLOPS)")
            if summary.pruned_strategies:
                print("  Pruned:")
                for p in summary.pruned_strategies:
                    print(f"    {p.get('id')} {p.get('name')} — {p.get('reason')}")
            if summary.report_path:
                print(f"  Report: {summary.report_path}")
            print("=" * 60)

    # ── Helpers ───────────────────────────────────────────────────────────

    def _print_header(self) -> None:
        t = self.tracker
        title = f"TECO: Optimizing {t.run_id.split('_', 3)[-1] if '_' in t.run_id else t.run_id}"
        hw = (
            f"Hardware: {t.device_name} | "
            f"Peak FP16: {t.peak_tflops_fp16:.0f} TFLOPS | "
            f"Peak BW: {t.peak_bandwidth_gbs:.0f} GB/s"
        )
        if HAS_RICH and self._console:
            self._console.rule(f"[bold cyan]{title}[/bold cyan]")
            self._console.print(f"[dim]{hw}[/dim]")
        else:
            print(f"\n━━━ {title} ━━━")
            print(hw)

    def _print_phase_banner(self, phase: str, message: str) -> None:
        label = f"── {phase} {'─' * max(0, 50 - len(phase))}"
        if message:
            label += f"  {message}"
        if HAS_RICH and self._console:
            self._console.print(f"\n[bold cyan]{label}[/bold cyan]")
        else:
            print(f"\n{label}")

    def _print(self, text: str, style: str | None = None) -> None:
        if HAS_RICH and self._console and style:
            self._console.print(f"[{style}]{text}[/{style}]")
        elif HAS_RICH and self._console:
            self._console.print(text)
        else:
            print(text)

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        if minutes < 60:
            return f"{minutes}m {secs:.0f}s"
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h {mins}m {secs:.0f}s"
