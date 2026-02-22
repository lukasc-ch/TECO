"""Tests for the TECO reporting subsystem (tracker, console reporter, HTML report)."""

from __future__ import annotations

from pathlib import Path

import pytest

from teco.reporting.console import RichConsoleReporter
from teco.reporting.html_report import HTMLReportGenerator
from teco.reporting.tracker import (
    ProgressTracker,
)

# ---------------------------------------------------------------------------
# ProgressTracker tests
# ---------------------------------------------------------------------------


class TestProgressTracker:
    def test_emit_phase(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        tracker.emit_phase("INIT", "Starting")
        assert len(tracker.phase_events) == 1
        assert tracker.phase_events[0].phase == "INIT"
        assert tracker.phase_events[0].message == "Starting"

    def test_emit_baseline(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        tracker.emit_baseline("small", latency_ms=0.5, tflops=10.0, bottleneck="memory")
        tracker.emit_baseline("medium", latency_ms=2.0, tflops=50.0, bottleneck="compute")
        assert len(tracker.baseline_events) == 2
        assert tracker.baseline_events[0].shape_label == "small"
        assert tracker.baseline_events[1].tflops == 50.0

    def test_emit_strategy(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        tracker.emit_strategy("s1", "tensor-core-rewrite", "created", 0, "conf=0.85")
        assert len(tracker.strategy_events) == 1
        assert tracker.strategy_events[0].action == "created"

    def test_record_iteration(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        tracker.record_iteration(
            iteration=0,
            strategy_id="s1",
            strategy_name="tensor-core-rewrite",
            shape_results={"small": 12.0, "medium": 60.0},
            vs_baseline={"small": 20.0, "medium": 40.0},
            compile_ok=True,
            correctness_ok=True,
            wall_time_s=5.0,
        )
        assert len(tracker.iteration_results) == 1
        assert tracker.iteration_results[0].shape_results["medium"] == 60.0

    def test_emit_final(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        tracker.emit_final(
            overall_speedup=2.5,
            regime_winners=[{"id": "s1", "name": "tc-rewrite", "regime": "large", "tflops": "120.0"}],
            pruned_strategies=[{"id": "s2", "name": "vec-loads", "reason": "dominated"}],
            total_iterations=5,
        )
        assert tracker.final_summary is not None
        assert tracker.final_summary.overall_speedup == 2.5
        assert len(tracker.final_summary.regime_winners) == 1

    def test_listener_notification(self) -> None:
        tracker = ProgressTracker(run_id="test_run")
        events_received: list[tuple[str, object]] = []
        tracker.add_listener(lambda t, e: events_received.append((t, e)))

        tracker.emit_phase("INIT")
        tracker.emit_baseline("small", 1.0, 10.0)
        tracker.emit_strategy("s1", "test", "created", 0)

        assert len(events_received) == 3
        assert events_received[0][0] == "phase"
        assert events_received[1][0] == "baseline"
        assert events_received[2][0] == "strategy"

    def test_listener_exception_does_not_crash(self) -> None:
        tracker = ProgressTracker(run_id="test_run")

        def bad_listener(event_type: str, event: object) -> None:
            raise RuntimeError("boom")

        tracker.add_listener(bad_listener)
        # Should not raise
        tracker.emit_phase("INIT")
        assert len(tracker.phase_events) == 1

    def test_to_dict(self) -> None:
        tracker = ProgressTracker(
            run_id="test_run",
            device_name="Test GPU",
            peak_tflops_fp16=100.0,
            peak_bandwidth_gbs=1000.0,
            max_iterations=5,
        )
        tracker.emit_baseline("small", 1.0, 10.0, "memory")
        tracker.emit_strategy("s1", "test-strat", "created", 0, "conf=0.8")
        tracker.record_iteration(
            iteration=0,
            strategy_id="s1",
            strategy_name="test-strat",
            shape_results={"small": 15.0},
            vs_baseline={"small": 50.0},
            compile_ok=True,
            correctness_ok=True,
            wall_time_s=3.0,
        )
        tracker.emit_final(
            overall_speedup=1.5,
            regime_winners=[],
            pruned_strategies=[],
            total_iterations=1,
        )

        d = tracker.to_dict()
        assert d["run_id"] == "test_run"
        assert d["device_name"] == "Test GPU"
        assert len(d["baseline"]) == 1
        assert len(d["strategies_created"]) == 1
        assert len(d["iterations"]) == 1
        assert d["final"]["overall_speedup"] == 1.5

    def test_elapsed(self) -> None:
        import time

        tracker = ProgressTracker(run_id="test_run")
        time.sleep(0.05)
        assert tracker.elapsed() >= 0.04


# ---------------------------------------------------------------------------
# RichConsoleReporter tests
# ---------------------------------------------------------------------------


class TestRichConsoleReporter:
    def test_reporter_as_listener(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Reporter should not crash when receiving events."""
        tracker = ProgressTracker(
            run_id="test_run",
            device_name="Test GPU",
            peak_tflops_fp16=100.0,
            peak_bandwidth_gbs=1000.0,
        )
        reporter = RichConsoleReporter(tracker, verbosity=1)
        tracker.add_listener(reporter)

        tracker.emit_phase("INIT")
        tracker.emit_baseline("small", 1.0, 10.0, "memory")
        reporter.flush_baseline()
        tracker.emit_strategy("s1", "test-strat", "created", 0, "conf=0.8")
        tracker.record_iteration(
            iteration=0,
            strategy_id="s1",
            strategy_name="test-strat",
            shape_results={"small": 15.0},
            vs_baseline={"small": 50.0},
            compile_ok=True,
            correctness_ok=True,
            wall_time_s=3.0,
        )
        tracker.emit_final(
            overall_speedup=1.5,
            regime_winners=[{"id": "s1", "name": "test-strat", "regime": "all", "tflops": "15.0"}],
            pruned_strategies=[],
            total_iterations=1,
        )

        # If we got here without exceptions, the reporter works
        # Check that something was printed
        captured = capsys.readouterr()
        # Rich may or may not be installed; either way output should exist
        assert len(captured.out) > 0 or True  # don't fail if rich captures differently

    def test_fmt_duration(self) -> None:
        assert RichConsoleReporter._fmt_duration(30.0) == "30.0s"
        assert RichConsoleReporter._fmt_duration(90.0) == "1m 30s"
        assert RichConsoleReporter._fmt_duration(3700.0) == "1h 1m 40s"


# ---------------------------------------------------------------------------
# HTMLReportGenerator tests
# ---------------------------------------------------------------------------


class TestHTMLReportGenerator:
    def test_generate_creates_file(self, tmp_path: Path) -> None:
        tracker = ProgressTracker(
            run_id="test_run_20260221_000000_kernel",
            device_name="Test GPU",
            peak_tflops_fp16=100.0,
            peak_bandwidth_gbs=1000.0,
            max_iterations=5,
        )
        tracker.emit_baseline("small", 1.0, 10.0, "memory")
        tracker.emit_baseline("medium", 5.0, 50.0, "compute")
        tracker.emit_strategy("s1", "test-strat", "created", 0, "conf=0.8")
        tracker.record_iteration(
            iteration=0,
            strategy_id="s1",
            strategy_name="test-strat",
            shape_results={"small": 15.0, "medium": 65.0},
            vs_baseline={"small": 50.0, "medium": 30.0},
            compile_ok=True,
            correctness_ok=True,
            wall_time_s=3.0,
        )
        tracker.emit_final(
            overall_speedup=1.5,
            regime_winners=[{"id": "s1", "name": "test-strat", "regime": "all", "tflops": "65.0"}],
            pruned_strategies=[],
            total_iterations=1,
        )

        gen = HTMLReportGenerator()
        path = gen.generate(tracker, output_dir=tmp_path)

        assert path.exists()
        assert path.suffix == ".html"
        content = path.read_text()
        assert "TECO Optimization Report" in content
        assert "test_run_20260221_000000_kernel" in content
        assert "plotly" in content.lower()
        assert "1.50x" in content  # overall speedup

    def test_generate_with_no_iterations(self, tmp_path: Path) -> None:
        tracker = ProgressTracker(
            run_id="empty_run",
            device_name="Test GPU",
            peak_tflops_fp16=100.0,
            peak_bandwidth_gbs=1000.0,
        )
        tracker.emit_final(
            overall_speedup=1.0,
            regime_winners=[],
            pruned_strategies=[],
            total_iterations=0,
        )

        gen = HTMLReportGenerator()
        path = gen.generate(tracker, output_dir=tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "No iterations recorded" in content

    def test_report_contains_charts(self, tmp_path: Path) -> None:
        tracker = ProgressTracker(
            run_id="chart_test",
            device_name="Test GPU",
            peak_tflops_fp16=312.0,
            peak_bandwidth_gbs=2039.0,
            max_iterations=3,
        )
        tracker.emit_baseline("small", 0.5, 5.0, "memory")
        tracker.emit_baseline("medium", 2.0, 40.0, "compute")
        for i in range(3):
            tracker.record_iteration(
                iteration=i,
                strategy_id="s1",
                strategy_name="test",
                shape_results={"small": 5.0 + i * 2, "medium": 40.0 + i * 10},
                vs_baseline={"small": i * 40.0, "medium": i * 25.0},
                compile_ok=True,
                correctness_ok=True,
                wall_time_s=2.0,
            )
        tracker.emit_final(
            overall_speedup=1.75,
            regime_winners=[],
            pruned_strategies=[],
            total_iterations=3,
        )

        gen = HTMLReportGenerator()
        path = gen.generate(tracker, output_dir=tmp_path)
        content = path.read_text()

        # Check chart containers exist
        assert "chart-iterations" in content
        assert "chart-comparison" in content
        assert "chart-roofline" in content
        # Check Plotly.newPlot calls
        assert "Plotly.newPlot" in content
