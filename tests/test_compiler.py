"""Tests for compiler and correctness validation utilities (no GPU required)."""

from __future__ import annotations

import pytest

from teco.tools.compiler import (
    _extract_kernel_section,
    _extract_test_section,
    _namespace_functions,
    _strip_module_level_statements,
    compile_check,
    extract_source_hash,
    validate_correctness,
)
from teco.tools.code_editor import (
    apply_replacement,
    apply_unified_diff,
    generate_diff,
)

# ---------------------------------------------------------------------------
# Simple kernel fixture (no Triton imports to keep tests fast)
# ---------------------------------------------------------------------------

_SIMPLE_KERNEL = """\
import torch

def add_kernel(a, b):
    return a + b

def wrapper(n):
    a = torch.ones(n, device='cpu')
    b = torch.ones(n, device='cpu')
    return add_kernel(a, b)

""" + "#" * 146 + """

def test_add_kernel():
    results = {}
    results['test_case_1'] = wrapper(16)
    results['test_case_2'] = wrapper(64)
    return results

result_gold = test_add_kernel()
"""

_SIMPLE_KERNEL_NO_SEP = """\
import torch

def add_kernel(a, b):
    return a + b

def test_add():
    return {"out": add_kernel(torch.ones(4), torch.ones(4))}
"""


class TestSourceExtraction:
    def test_extract_kernel_section_with_separator(self) -> None:
        kernel = _extract_kernel_section(_SIMPLE_KERNEL)
        assert "def add_kernel" in kernel
        assert "def test_add_kernel" not in kernel

    def test_extract_test_section_with_separator(self) -> None:
        test = _extract_test_section(_SIMPLE_KERNEL)
        assert "def test_add_kernel" in test

    def test_extract_kernel_section_no_separator(self) -> None:
        kernel = _extract_kernel_section(_SIMPLE_KERNEL_NO_SEP)
        assert "def add_kernel" in kernel

    def test_extract_test_section_no_separator(self) -> None:
        test = _extract_test_section(_SIMPLE_KERNEL_NO_SEP)
        assert "def test_add" in test

    def test_namespace_functions(self) -> None:
        source = "def test_foo():\n    pass\ndef test_bar():\n    pass\n"
        result = _namespace_functions(source, prefix="cand_")
        assert "def cand_test_foo" in result
        assert "def cand_test_bar" in result
        assert "def test_foo" not in result

    def test_strip_keeps_defs_and_imports(self) -> None:
        source = (
            "import torch\n"
            "from pathlib import Path\n"
            "\n"
            "def test_foo():\n"
            "    return {'x': 1}\n"
            "\n"
            "result_gold = test_foo()\n"
            "print(result_gold)\n"
        )
        stripped = _strip_module_level_statements(source)
        assert "import torch" in stripped
        assert "from pathlib" in stripped
        assert "def test_foo" in stripped
        assert "result_gold" not in stripped
        assert "print(result_gold)" not in stripped

    def test_strip_leaves_source_unchanged_if_no_side_effects(self) -> None:
        # Blank lines are not AST nodes and may be dropped; only substance must be preserved.
        source = "import torch\n\ndef test_foo():\n    pass\n"
        stripped = _strip_module_level_statements(source)
        assert "import torch" in stripped
        assert "def test_foo" in stripped

    def test_strip_invalid_syntax_passthrough(self) -> None:
        bad = "def foo(\n"
        assert _strip_module_level_statements(bad) == bad


class TestCompileCheck:
    def test_valid_source(self) -> None:
        result = compile_check("x = 1 + 1\n")
        assert result.success

    def test_syntax_error(self) -> None:
        result = compile_check("def foo(\n")
        assert not result.success
        assert "SyntaxError" in result.error_message

    def test_source_hash_consistent(self) -> None:
        h1 = extract_source_hash("hello world")
        h2 = extract_source_hash("hello world")
        assert h1 == h2

    def test_source_hash_differs(self) -> None:
        assert extract_source_hash("aaa") != extract_source_hash("bbb")


class TestCodeEditor:
    def test_apply_replacement_basic(self) -> None:
        original = "def foo():\n    return 1\n"
        result = apply_replacement(original, "return 1", "return 2")
        assert result.success
        assert "return 2" in result.patched_source

    def test_apply_replacement_not_found(self) -> None:
        original = "def foo():\n    return 1\n"
        result = apply_replacement(original, "return 999", "return 0")
        assert not result.success

    def test_generate_diff_produces_unified_format(self) -> None:
        original = "line1\nline2\nline3\n"
        modified = "line1\nline2_modified\nline3\n"
        diff = generate_diff(original, modified)
        assert "---" in diff
        assert "+++" in diff
        assert "-line2\n" in diff
        assert "+line2_modified\n" in diff

    def test_apply_unified_diff_roundtrip(self) -> None:
        original = "def foo():\n    x = 1\n    return x\n"
        modified = "def foo():\n    x = 2\n    return x\n"
        diff = generate_diff(original, modified)
        result = apply_unified_diff(original, diff)
        assert result.success
        assert "x = 2" in result.patched_source


# ---------------------------------------------------------------------------
# validate_correctness (CPU-only, no Triton)
# ---------------------------------------------------------------------------

# A minimal TritonBench-style kernel file: tensor output
_SEP = "#" * 146

_KERNEL_TENSOR = f"""\
import torch

def compute(x):
    return x * 2

{_SEP}

import torch

def test_compute():
    results = {{"test_case_1": compute(torch.ones(4))}}
    return results

result_gold = test_compute()
"""

# TritonBench-style: dict values are tuples of numpy arrays (like lightning_attention)
_KERNEL_TUPLE_NUMPY = f"""\
import torch
import numpy as np

def compute(x):
    return x * 2

{_SEP}

import torch
import numpy as np

def test_compute():
    out = compute(torch.ones(4))
    return {{"test_case_1": (out.numpy(), out.numpy())}}

result_gold = test_compute()
print(result_gold)
"""


class TestValidateCorrectness:
    def test_identical_sources_pass(self) -> None:
        result = validate_correctness(_KERNEL_TENSOR, _KERNEL_TENSOR)
        assert result.success, result.error_message

    def test_wrong_output_fails(self) -> None:
        wrong = _KERNEL_TENSOR.replace("x * 2", "x * 3")
        result = validate_correctness(_KERNEL_TENSOR, wrong)
        assert not result.success
        assert "MISMATCH" in result.error_message

    def test_tuple_numpy_outputs_pass(self) -> None:
        result = validate_correctness(_KERNEL_TUPLE_NUMPY, _KERNEL_TUPLE_NUMPY)
        assert result.success, result.error_message

    def test_module_level_side_effects_ignored(self) -> None:
        # result_gold = ... and print(...) at module level must not cause failures
        result = validate_correctness(_KERNEL_TUPLE_NUMPY, _KERNEL_TUPLE_NUMPY)
        assert result.success, result.error_message

    def test_name_collision_isolation(self) -> None:
        # Candidate redefines 'compute' with a different result — should FAIL comparison,
        # not crash, proving namespaces are isolated.
        cand = _KERNEL_TENSOR.replace("x * 2", "x * 99")
        result = validate_correctness(_KERNEL_TENSOR, cand)
        assert not result.success
        assert "MISMATCH" in result.error_message
