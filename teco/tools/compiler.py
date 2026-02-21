"""Kernel compilation and correctness validation tool."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
import subprocess
import sys
import tempfile
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
    """Build a self-contained Python script that validates candidate against reference.

    Each source is exec'd in its own namespace dict so kernel function names in the
    candidate never overwrite those in the reference.  The test function found in each
    namespace calls its own kernel via its own __globals__, giving true isolation without
    any renaming heuristics.

    Comparison is recursive so it handles dict values that are tensors, numpy arrays,
    or arbitrarily-nested tuples/lists of the above (all common in TritonBench tests).
    """
    ref_kernel = _extract_kernel_section(reference_source)
    ref_test = _strip_module_level_statements(_extract_test_section(reference_source))
    cand_kernel = _extract_kernel_section(candidate_source)
    cand_test = _strip_module_level_statements(_extract_test_section(candidate_source))

    ref_full = ref_kernel + "\n" + ref_test
    cand_full = cand_kernel + "\n" + cand_test

    lines = [
        "import sys",
        "import io",
        "import os",
        "import importlib.util",
        "import tempfile",
        "import torch",
        "try:",
        "    import numpy as _np",
        "except ImportError:",
        "    _np = None",
        "",
        f"_ATOL = {atol}",
        f"_RTOL = {rtol}",
        f"_REF_SOURCE = {repr(ref_full)}",
        f"_CAND_SOURCE = {repr(cand_full)}",
        "",
        "def _to_tensor(x):",
        "    if _np is not None and isinstance(x, _np.ndarray):",
        "        return torch.from_numpy(x)",
        "    return x",
        "",
        "def _compare(r, c, path=''):",
        "    r, c = _to_tensor(r), _to_tensor(c)",
        "    if isinstance(r, torch.Tensor):",
        "        if not isinstance(c, torch.Tensor):",
        "            raise AssertionError(f'Type mismatch at {path!r}: ref={type(r).__name__}, cand={type(c).__name__}')",
        "        if not torch.allclose(r.float(), c.float(), atol=_ATOL, rtol=_RTOL):",
        "            diff = (r.float() - c.float()).abs().max().item()",
        "            raise AssertionError(f'MISMATCH at {path!r}: max_diff={diff:.6f}')",
        "    elif isinstance(r, (list, tuple)):",
        "        for i, (ri, ci) in enumerate(zip(r, c)):",
        "            _compare(ri, ci, f'{path}[{i}]')",
        "    # scalars / other non-tensor types: skip numeric comparison",
        "",
        # Load source from a real .py file so Triton @jit can read it via inspect.
        # Stdout is redirected during exec_module to swallow TritonBench's
        # module-level  result_gold = test_...()  / print(result_gold) calls.
        "def _load_module(source, name):",
        "    f = tempfile.NamedTemporaryFile(suffix='.py', mode='w', delete=False)",
        "    f.write(source)",
        "    f.close()",
        "    spec = importlib.util.spec_from_file_location(name, f.name)",
        "    mod = importlib.util.module_from_spec(spec)",
        "    old = sys.stdout",
        "    sys.stdout = io.StringIO()",
        "    try:",
        "        spec.loader.exec_module(mod)",
        "    finally:",
        "        sys.stdout = old",
        "    return mod, f.name",
        "",
        "_ref_path = _cand_path = None",
        "try:",
        "    _ref_mod, _ref_path = _load_module(_REF_SOURCE, 'ref_kernel')",
        "    _cand_mod, _cand_path = _load_module(_CAND_SOURCE, 'cand_kernel')",
        "    _ref_fn = next((v for k, v in vars(_ref_mod).items() if k.startswith('test_') and callable(v)), None)",
        "    _cand_fn = next((v for k, v in vars(_cand_mod).items() if k.startswith('test_') and callable(v)), None)",
        "    if _ref_fn is None or _cand_fn is None:",
        "        print('ERROR: Could not find test functions', file=sys.stderr)",
        "        sys.exit(1)",
        "    _ref_out = _ref_fn()",
        "    _cand_out = _cand_fn()",
        "    if not isinstance(_ref_out, dict) or not isinstance(_cand_out, dict):",
        "        print('ERROR: test functions must return dict', file=sys.stderr)",
        "        sys.exit(1)",
        "    for _k in _ref_out:",
        "        if _k not in _cand_out:",
        "            print(f'ERROR: candidate missing key {_k!r}', file=sys.stderr)",
        "            sys.exit(1)",
        "        _compare(_ref_out[_k], _cand_out[_k], _k)",
        "    print('OK')",
        "except SystemExit:",
        "    raise",
        "except Exception as _e:",
        "    import traceback",
        "    print(f'ERROR: {_e}', file=sys.stderr)",
        "    traceback.print_exc(file=sys.stderr)",
        "    sys.exit(1)",
        "finally:",
        "    for _p in (_ref_path, _cand_path):",
        "        if _p is not None:",
        "            try: os.unlink(_p)",
        "            except OSError: pass",
        "",
    ]
    return "\n".join(lines) + "\n"


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


def _strip_module_level_statements(source: str) -> str:
    """Remove module-level execution statements, keeping only definitions and imports.

    TritonBench files always end the test section with lines like:
        result_gold = test_...()
        print(result_gold)
    These must NOT run at module-load time because:
    - They invoke the kernel (potentially OOM on restricted hardware)
    - They execute twice (once at load, once when we call _ref_fn())

    Uses AST to identify which lines belong to top-level function/class/import nodes
    and discards everything else (bare expressions, assignments, etc.).
    Falls back to the original source if parsing fails.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source

    lines = source.splitlines(keepends=True)
    keep: set[int] = set()
    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.Import,
                ast.ImportFrom,
            ),
        ):
            for lineno in range(node.lineno, node.end_lineno + 1):
                keep.add(lineno)  # 1-indexed

    return "".join(line for i, line in enumerate(lines, 1) if i in keep)
