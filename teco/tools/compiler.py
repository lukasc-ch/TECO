"""Kernel compilation and correctness validation tool."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
import textwrap
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# Delimiter used in TritonBench files to separate kernel code from tests
_TEST_SEPARATOR = "#" * 146


@dataclass
class ValidationResult:
    success: bool
    error_message: str = ""
    # Outputs from the test function, keyed by test case name
    outputs: dict[str, Any] | None = None


@dataclass
class CompileResult:
    success: bool
    error_message: str = ""
    source_hash: str = ""


# ---------------------------------------------------------------------------
# Compilation check
# ---------------------------------------------------------------------------


def compile_check(kernel_source: str) -> CompileResult:
    """
    Attempt to parse and import a kernel file to check for syntax/import errors.
    Does not execute the kernel — just imports it.

    Returns CompileResult with success status and any error message.
    """
    source_hash = hashlib.sha256(kernel_source.encode()).hexdigest()[:16]

    # Syntax check via ast
    try:
        ast.parse(kernel_source)
    except SyntaxError as e:
        return CompileResult(
            success=False,
            error_message=f"SyntaxError at line {e.lineno}: {e.msg}",
            source_hash=source_hash,
        )

    # Import check via subprocess (isolates CUDA context)
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        # Write only the kernel portion (before test separator) for import check
        kernel_only = _extract_kernel_section(kernel_source)
        f.write(kernel_only)
        tmp_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, "-c", f"import importlib.util; "
             f"spec = importlib.util.spec_from_file_location('kernel', '{tmp_path}'); "
             f"mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            return CompileResult(
                success=False,
                error_message=result.stderr.strip()[:2000],
                source_hash=source_hash,
            )
    except subprocess.TimeoutExpired:
        return CompileResult(
            success=False,
            error_message="Import timed out (60s)",
            source_hash=source_hash,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    return CompileResult(success=True, source_hash=source_hash)


# ---------------------------------------------------------------------------
# Correctness validation
# ---------------------------------------------------------------------------


def validate_correctness(
    reference_source: str,
    candidate_source: str,
    atol: float = 1e-3,
    rtol: float = 1e-3,
    timeout: int = 120,
) -> ValidationResult:
    """
    Run the test function from both reference and candidate kernels,
    compare tensor outputs with torch.allclose.

    Supports the TritonBench format: kernel code separated from test by 146 '#' chars.
    The test function must return a dict[str, torch.Tensor].

    Args:
        reference_source: Source of the reference kernel (original).
        candidate_source: Source of the candidate kernel (optimized).
        atol: Absolute tolerance for torch.allclose.
        rtol: Relative tolerance for torch.allclose.
        timeout: Subprocess timeout in seconds.

    Returns:
        ValidationResult with success=True if all outputs match.
    """
    validation_script = _build_validation_script(
        reference_source, candidate_source, atol=atol, rtol=rtol
    )

    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(validation_script)
        script_path = Path(f.name)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return ValidationResult(success=False, error_message=f"Validation timed out ({timeout}s)")
    finally:
        script_path.unlink(missing_ok=True)

    if result.returncode == 0:
        return ValidationResult(success=True)
    else:
        return ValidationResult(
            success=False,
            error_message=(result.stdout + result.stderr).strip()[:3000],
        )


def _build_validation_script(
    reference_source: str,
    candidate_source: str,
    atol: float,
    rtol: float,
) -> str:
    """Build a self-contained Python script that validates candidate against reference."""
    ref_kernel = _extract_kernel_section(reference_source)
    ref_test = _extract_test_section(reference_source)
    cand_kernel = _extract_kernel_section(candidate_source)
    cand_test = _extract_test_section(candidate_source)

    # Rename functions in candidate to avoid collisions
    cand_kernel_renamed = _namespace_functions(cand_kernel, prefix="cand_")
    cand_test_renamed = _namespace_functions(cand_test, prefix="cand_")

    return textwrap.dedent(f"""\
        import torch
        import sys

        # ---- Reference kernel ----
        {textwrap.indent(ref_kernel, '        ').strip()}

        # ---- Reference test ----
        {textwrap.indent(ref_test, '        ').strip()}

        # ---- Candidate kernel (namespaced) ----
        {textwrap.indent(cand_kernel_renamed, '        ').strip()}

        # ---- Candidate test (namespaced) ----
        {textwrap.indent(cand_test_renamed, '        ').strip()}

        # ---- Validation ----
        try:
            ref_fn = [v for k, v in list(locals().items()) + list(globals().items())
                      if k.startswith('test_') and callable(v) and not k.startswith('test_cand_')]
            cand_fn = [v for k, v in list(locals().items()) + list(globals().items())
                       if k.startswith('cand_test_') and callable(v)]
            if not ref_fn or not cand_fn:
                print("ERROR: Could not find test functions", file=sys.stderr)
                sys.exit(1)
            ref_out = ref_fn[0]()
            cand_out = cand_fn[0]()
            if not isinstance(ref_out, dict) or not isinstance(cand_out, dict):
                print("ERROR: test functions must return dict[str, Tensor]", file=sys.stderr)
                sys.exit(1)
            for key in ref_out:
                r = ref_out[key]
                c = cand_out.get(key)
                if c is None:
                    print(f"ERROR: candidate missing output key '{{key}}'", file=sys.stderr)
                    sys.exit(1)
                if not torch.allclose(r.float(), c.float(), atol={atol}, rtol={rtol}):
                    max_diff = (r.float() - c.float()).abs().max().item()
                    print(f"MISMATCH key='{{key}}' max_diff={{max_diff:.6f}} "
                          f"atol={atol} rtol={rtol}", file=sys.stderr)
                    sys.exit(1)
            print("OK")
        except Exception as e:
            print(f"ERROR: {{e}}", file=sys.stderr)
            sys.exit(1)
    """)


# ---------------------------------------------------------------------------
# Source manipulation utilities
# ---------------------------------------------------------------------------


def extract_source_hash(source: str) -> str:
    return hashlib.sha256(source.encode()).hexdigest()


def _extract_kernel_section(source: str) -> str:
    """Return the kernel code portion (before the test separator, if present)."""
    if _TEST_SEPARATOR in source:
        return source.split(_TEST_SEPARATOR)[0].rstrip()
    return source


def _extract_test_section(source: str) -> str:
    """Return the test function portion (after the test separator, if present)."""
    if _TEST_SEPARATOR in source:
        parts = source.split(_TEST_SEPARATOR, 1)
        return parts[1].lstrip("\n") if len(parts) > 1 else ""
    # Fallback: look for any function named test_*
    lines = source.splitlines()
    test_start = next(
        (i for i, l in enumerate(lines) if re.match(r"^def test_", l)), None
    )
    if test_start is not None:
        return "\n".join(lines[test_start:])
    return ""


def _namespace_functions(source: str, prefix: str) -> str:
    """Prefix all top-level function definitions with `prefix`."""
    return re.sub(r"^def (test_)", f"def {prefix}\\1", source, flags=re.MULTILINE)
