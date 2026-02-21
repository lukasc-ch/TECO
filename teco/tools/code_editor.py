"""Code editing tool: apply unified diffs and manage kernel source files."""

from __future__ import annotations

import difflib
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PatchResult:
    success: bool
    patched_source: str = ""
    error_message: str = ""
    unified_diff: str = ""  # diff between original and result


def apply_unified_diff(original_source: str, diff_text: str) -> PatchResult:
    """
    Apply a unified diff (patch) to the original source.

    Accepts diffs in standard unified diff format (--- / +++ / @@ headers).
    Falls back to fuzzy matching if exact context doesn't match.

    Returns PatchResult with patched source or error.
    """
    # Try subprocess patch first (most reliable for well-formed diffs)
    result = _apply_via_patch(original_source, diff_text)
    if result.success:
        return result

    # Fallback: apply via Python difflib hunk-by-hunk
    result = _apply_hunks_python(original_source, diff_text)
    return result


def apply_replacement(
    original_source: str,
    old_code: str,
    new_code: str,
    context_lines: int = 3,
) -> PatchResult:
    """
    Replace a code block by exact string match.
    More robust than diff when the LLM provides before/after blocks.

    Args:
        original_source: Full source code.
        old_code: Exact code to find and replace.
        new_code: Replacement code.
        context_lines: Lines of context in the output diff.

    Returns:
        PatchResult with patched source.
    """
    # Normalize indentation for matching (strip common leading whitespace)
    norm_original = original_source
    norm_old = old_code.strip()

    if norm_old not in norm_original:
        # Try ignoring trailing whitespace on each line
        stripped_original = "\n".join(l.rstrip() for l in norm_original.splitlines())
        stripped_old = "\n".join(l.rstrip() for l in norm_old.splitlines())
        if stripped_old not in stripped_original:
            return PatchResult(
                success=False,
                error_message="Could not find old_code block in source (exact match failed).",
            )
        norm_original = stripped_original
        norm_old = stripped_old

    patched = norm_original.replace(norm_old, new_code.strip(), 1)
    diff = _make_diff(original_source, patched)
    return PatchResult(success=True, patched_source=patched, unified_diff=diff)


def generate_diff(original: str, modified: str, fromfile: str = "original", tofile: str = "modified") -> str:
    """Generate a unified diff string between two source strings."""
    return _make_diff(original, modified, fromfile=fromfile, tofile=tofile)


def write_kernel_file(path: Path, source: str) -> None:
    """Atomically write kernel source to a file."""
    tmp = path.with_suffix(".py.tmp")
    tmp.write_text(source, encoding="utf-8")
    tmp.replace(path)


def read_kernel_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_via_patch(original_source: str, diff_text: str) -> PatchResult:
    """Apply diff using the system `patch` command."""
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as orig_f:
        orig_f.write(original_source)
        orig_path = Path(orig_f.name)
    with tempfile.NamedTemporaryFile(suffix=".patch", mode="w", delete=False) as diff_f:
        diff_f.write(diff_text)
        diff_path = Path(diff_f.name)

    try:
        result = subprocess.run(
            ["patch", "--quiet", "--force", str(orig_path), str(diff_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            patched = orig_path.read_text()
            diff = _make_diff(original_source, patched)
            return PatchResult(success=True, patched_source=patched, unified_diff=diff)
        else:
            return PatchResult(
                success=False,
                error_message=f"patch failed: {result.stdout} {result.stderr}".strip()[:1000],
            )
    except FileNotFoundError:
        return PatchResult(success=False, error_message="`patch` command not found")
    except subprocess.TimeoutExpired:
        return PatchResult(success=False, error_message="`patch` timed out")
    finally:
        orig_path.unlink(missing_ok=True)
        diff_path.unlink(missing_ok=True)
        # patch creates a .orig backup
        backup = orig_path.with_suffix(".py.orig")
        backup.unlink(missing_ok=True)


def _apply_hunks_python(original_source: str, diff_text: str) -> PatchResult:
    """
    Pure-Python hunk-by-hunk diff application.
    Handles basic unified diffs; tolerates some context drift.
    """
    lines = original_source.splitlines(keepends=True)
    hunks = _parse_hunks(diff_text)
    if not hunks:
        return PatchResult(
            success=False, error_message="No valid hunks found in diff."
        )

    offset = 0  # cumulative line count shift
    for hunk_start, removals, additions in hunks:
        target_line = hunk_start - 1 + offset  # 0-indexed

        # Fuzzy search: find the best match for the removal context
        context = [l for l in removals if not l.startswith("-")]
        match_line = _find_context(lines, target_line, context)
        if match_line is None:
            return PatchResult(
                success=False,
                error_message=f"Could not find hunk context near line {hunk_start}.",
            )

        # Compute actual extent of removal
        remove_lines = [l[1:] for l in removals if l.startswith("-") or not l.startswith("+")]
        n_remove = sum(1 for l in removals if l.startswith("-") or not l.startswith("+"))
        add_lines = [l[1:] if l.startswith("+") else l[1:] for l in additions if l.startswith("+")]

        lines[match_line: match_line + n_remove] = [
            (l if l.endswith("\n") else l + "\n") for l in add_lines
        ]
        offset += len(add_lines) - n_remove

    patched = "".join(lines)
    diff = _make_diff(original_source, patched)
    return PatchResult(success=True, patched_source=patched, unified_diff=diff)


def _parse_hunks(diff_text: str) -> list[tuple[int, list[str], list[str]]]:
    """Parse unified diff into (start_line, context+removals, additions) tuples."""
    hunks = []
    current_start = 0
    current_lines: list[str] = []
    in_hunk = False

    for line in diff_text.splitlines():
        m = re.match(r"^@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
        if m:
            if in_hunk and current_lines:
                removals = [l for l in current_lines if not l.startswith("+")]
                additions = [l for l in current_lines if not l.startswith("-")]
                hunks.append((current_start, removals, additions))
            current_start = int(m.group(1))
            current_lines = []
            in_hunk = True
        elif in_hunk and (line.startswith(" ") or line.startswith("+") or line.startswith("-")):
            current_lines.append(line)

    if in_hunk and current_lines:
        removals = [l for l in current_lines if not l.startswith("+")]
        additions = [l for l in current_lines if not l.startswith("-")]
        hunks.append((current_start, removals, additions))

    return hunks


def _find_context(lines: list[str], hint: int, context: list[str], window: int = 20) -> int | None:
    """Find the best matching line for context near hint (0-indexed). Returns line index or None."""
    if not context:
        return hint

    context_clean = [l.strip() for l in context if l.strip()]
    if not context_clean:
        return hint

    best_ratio = 0.0
    best_line = None
    start = max(0, hint - window)
    end = min(len(lines), hint + window + len(context_clean))

    for i in range(start, end):
        segment = [lines[j].rstrip() for j in range(i, min(i + len(context_clean), len(lines)))]
        ratio = difflib.SequenceMatcher(None, context_clean, segment).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_line = i

    return best_line if best_ratio >= 0.6 else None


def _make_diff(
    original: str,
    modified: str,
    fromfile: str = "original",
    tofile: str = "modified",
) -> str:
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(keepends=True),
            modified.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )
    return "".join(diff_lines)
