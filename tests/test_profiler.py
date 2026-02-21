"""Tests for profiler utilities (no GPU required for most)."""

from __future__ import annotations

from teco.tools.profiler import (
    _lookup_peak_bandwidth,
    _lookup_peak_tflops,
    _parse_ncu_csv,
    _safe_ratio,
)


class TestHardwareLookup:
    def test_a100_bandwidth(self) -> None:
        bw = _lookup_peak_bandwidth("NVIDIA A100-SXM4-80GB")
        assert bw == 2039.0

    def test_h100_tflops(self) -> None:
        tflops = _lookup_peak_tflops("NVIDIA H100 SXM5 80GB", "fp16")
        assert tflops == 989.0

    def test_unknown_device_fallback(self) -> None:
        bw = _lookup_peak_bandwidth("SomeUnknownGPU-XYZ")
        assert bw > 0  # fallback value

    def test_unknown_dtype_fallback(self) -> None:
        tflops = _lookup_peak_tflops("NVIDIA A100-SXM4-80GB", "bf16")
        # bf16 not in table, should fall back to fp16 or conservative default
        assert tflops > 0


class TestSafeRatio:
    def test_normal(self) -> None:
        assert _safe_ratio(10.0, 2.0) == 5.0

    def test_zero_denominator(self) -> None:
        assert _safe_ratio(10.0, 0.0) == 0.0

    def test_zero_numerator(self) -> None:
        assert _safe_ratio(0.0, 5.0) == 0.0
