# TECO — The Experienced Code Optimizer: Implementation Plan

## Context

TECO is an agentic GPU kernel optimizer that improves performance of code (starting with Triton)
by combining LLM reasoning with real profiling feedback. Its distinguishing feature is
**learned experience**: it behaves like a skilled student — studying documentation, running
targeted experiments, and reflecting to extract generalizable lessons after each optimization run.

This plan covers the **MVP phase**: the core optimizer loop with local Triton execution and the
structural skeleton for the full multi-agent system.

---

## Architecture

### Agent framework: smolagents
- `smolagents` (Hugging Face) drives each agent via tool use
- Each agent is a `CodeAgent` or `ToolCallingAgent` with a curated set of tools
- Claude (via Anthropic SDK) is the backing LLM

### Agent roles (all built, only Optimizer active in MVP)
| Agent | Role | MVP status |
|---|---|---|
| `LearnerAgent` | Crawl docs/papers for language + hardware | Stub only |
| `OptimizerAgent` | Profile → hypothesize → apply → re-profile loop | **Full** |
| `ExperimenterAgent` | Design microbenchmarks for ambiguous bottlenecks | Stub only |
| `ReflectorAgent` | Extract generalizable patterns post-run | Stub only |

### Knowledge store: hierarchical Markdown + JSON
```
knowledge/
  runs/           # One .md per optimization run (input, steps, outcome)
  patterns/       # Generalizable rules (e.g., block_size_triton_a100.md)
  lessons/        # High-level synthesized insights across runs
```
Each run file is structured (YAML front-matter + Markdown body) so it can be
parsed by the Reflector later without a database.

---

## Project structure

```
teco/
  agents/
    base.py            # BaseAgent + shared utilities
    learner.py         # LearnerAgent (stub)
    optimizer.py       # OptimizerAgent — FULL MVP
    experimenter.py    # ExperimenterAgent (stub)
    reflector.py       # ReflectorAgent (stub)
  tools/
    profiler.py        # Two-stage profiling: do_bench + ncu
    compiler.py        # compile_and_validate() — subprocess Triton compilation
    code_editor.py     # apply_patch() — write/diff kernel files
    search.py          # web_search() + doc_fetch() (stub)
  knowledge/
    store.py           # KnowledgeStore: read/write runs/, patterns/, lessons/
    schema.py          # Pydantic models: RunRecord, Pattern, Lesson, Strategy, StrategyTree
  orchestration/
    loop.py            # Main optimization loop: init → strategy planning → iteration → finalise
    context.py         # OptimizationContext: task, hardware, StrategyTree, iteration state
  cli.py               # CLI entry point (typer)
pyproject.toml
tests/
  test_profiler.py
  test_knowledge_store.py
  test_optimizer_loop.py     # Integration: runs on TritonBench entry
```

---

## Critical implementation details

### OptimizerAgent loop (core MVP)

The loop has two distinct phases per round: **strategy planning** (beam-search style branching)
and **refinement** (deepening the best active branch per shape regime).

Key insight: a strategy that is optimal for large batch sizes may be suboptimal for small ones.
Strategies are therefore never globally pruned solely because another strategy wins on some shapes —
they are pruned only when their projected ceiling is dominated **across all tested shape regimes**.
Multiple strategies can co-exist as winners for disjoint shape ranges, and are combined into a
single dispatch wrapper in the final output.

```
── INIT ────────────────────────────────────────────────────────────────────
1. Load kernel + parse into OptimizationContext
2. Identify the shape parameter space: extract all variable dims + tuneable params
   (e.g., M, N, K, batch, head_dim). Define a representative shape sweep
   (small / medium / large for each axis) used throughout profiling.
3. Profile baseline across shape sweep: Stage 1 (latency) + Stage 2 (ncu deep)
   → one ProfilingReport per shape point
4. Query KnowledgeStore for similar past strategies and shape-dependent outcomes

── STRATEGY PLANNING (once, before iteration loop) ─────────────────────────
5. LLM thinking call: given (code, shape-sweep ProfilingReports, past outcomes)
   → generate N=3–5 fundamentally different high-level strategies
   Each strategy has:
     - name + rationale
     - expected bottleneck it addresses
     - rough implementation sketch
     - predicted winning shape regime (e.g., "likely best for M > 1024")
     - incompatibility note (why it precludes other strategies in the same regime)
   Store as StrategyTree in OptimizationContext

── ITERATION LOOP ──────────────────────────────────────────────────────────
Each iteration:
  6. PLAN REVIEW: LLM re-scores all active strategies per shape regime
     given accumulated profiling data across the shape sweep
     → prune a strategy only if it is dominated in ALL shape regimes by others
     → mark strategies as "regime winner" if they lead for a contiguous shape range
  7. DEEPEN: For each still-active strategy, produce a concrete code diff
     (strategies in disjoint winning regimes are deepened in parallel)
  8. Apply diffs → compile → validate correctness for each
  9. Stage 1 profile across shape sweep → update per-strategy, per-shape performance
  10. Repeat until: (a) max iterations, (b) all strategies have stable regime assignments,
      or (c) all regimes within X% of hardware ceiling

── FINALISE ────────────────────────────────────────────────────────────────
11. Stage 2 (ncu) on each regime-winning kernel → full ProfilingReport per strategy
12. LLM synthesis call: given winning strategies + their shape regimes
    → generate a dispatch wrapper that selects the right implementation at runtime
    based on input shape/param heuristics
13. Validate dispatch wrapper: run full shape sweep, confirm each path hits expected perf
14. Trigger ReflectorAgent stub:
    - Writes run .md with all strategies, shape-regime assignments, and outcomes
    - Logs ALL strategies (including pruned) with shape context to KnowledgeStore
```

**Data models:**
```python
class ShapePoint(BaseModel):
    dims: dict[str, int]               # e.g., {"M": 512, "N": 1024, "K": 64}
    label: str                         # "small" | "medium" | "large" | custom

class StrategyShapeResult(BaseModel):
    shape_point: ShapePoint
    achieved_tflops: float
    profiling_report: ProfilingReport | None  # None for Stage-1-only iterations

class Strategy(BaseModel):
    id: str
    name: str                          # e.g., "tensor-core-rewrite"
    rationale: str
    addresses_bottleneck: str
    incompatible_with: list[str]       # ids of strategies incompatible in same regime
    predicted_winning_regime: str      # LLM free-text, e.g. "M > 1024, power-of-two N"
    llm_confidence: float              # 0–1, updated each iteration
    status: Literal["active", "pruned", "regime_winner"]
    winning_shape_range: ShapeRange | None  # set when status == "regime_winner"
    prune_reason: str | None
    shape_results: list[StrategyShapeResult]
    current_code: str                  # latest implementation of this strategy
    iterations_applied: int

class ShapeRange(BaseModel):
    description: str                   # human-readable, e.g. "M <= 512"
    conditions: dict[str, str]         # parseable, e.g. {"M": "<= 512"}

class StrategyTree(BaseModel):
    strategies: list[Strategy]
    shape_sweep: list[ShapePoint]
    active_ids: list[str]
    regime_winners: dict[str, str]     # strategy_id → shape range description
    iteration: int
```

### Profiling tool (`teco/tools/profiler.py`)

Two-stage profiling:

Both stages run across the full **shape sweep** (small / medium / large representative points),
producing a `list[StrategyShapeResult]` rather than a single scalar.

**Stage 1 — Latency (fast, every iteration, all shape points):**
- `triton.testing.do_bench(fn, warmup=25, rep=100)` per shape → ms, derived GB/s + TFLOPS

**Stage 2 — Deep profiling (ncu, on baseline + final per winning strategy):**
- Invoke `ncu` via subprocess on a thin Python runner script
- Use `--set full` to capture all metric sections
- Parse CSV output (`--csv --page raw`) into structured `ProfilingReport`
- Run at the **most representative shape** for each strategy's winning regime

**Metrics extracted and surfaced to the agent:**

| Category | Metrics |
|---|---|
| Roofline | Achieved TFLOPS, arithmetic intensity, memory-bound vs compute-bound classification |
| Memory hierarchy | L1/L2 hit rates, DRAM bandwidth (GB/s), shared memory bank conflicts, global load/store efficiency |
| Warp efficiency | Warp execution efficiency, branch divergence %, instruction replay overhead, achieved vs theoretical occupancy |
| Instruction mix | FP32/FP16/INT op counts, tensor core utilization, special function unit usage |

**`ProfilingReport` Pydantic model:**
```python
class ProfilingReport(BaseModel):
    # Roofline
    achieved_tflops: float
    peak_tflops: float           # from pynvml
    arithmetic_intensity: float  # FLOP/byte
    bottleneck: Literal["memory", "compute", "latency"]

    # Memory
    l1_hit_rate: float
    l2_hit_rate: float
    dram_bandwidth_gbs: float
    peak_bandwidth_gbs: float    # from pynvml
    shared_mem_bank_conflicts: int

    # Warp
    warp_efficiency: float
    branch_divergence_pct: float
    achieved_occupancy: float
    theoretical_occupancy: float

    # Instruction mix
    tensor_core_utilization: float
    fp16_ops: int
    fp32_ops: int
    special_func_ops: int

    # Raw ncu CSV path for agent to request deeper drill-down
    raw_ncu_report_path: str
```

Hardware ceilings detected via `pynvml` at startup, stored in `OptimizationContext`.
System requirement: `ncu` (Nsight Compute) must be on PATH.

### Correctness validation
- Re-use TritonBench's test structure: extract `test_*()` function from kernel file
- Run reference (original) and candidate (optimized), compare tensor outputs
- Tolerance: `torch.allclose(ref, out, atol=1e-3, rtol=1e-3)`

### Knowledge store

#### Directory layout

```
knowledge/
  index.json                          # lightweight metadata cache (auto-maintained)
  runs/
    run_<YYYYMMDD>_<HHMMSS>_<slug>.json   # one per run — structured JSON
    ncu/                              # raw ncu report files (.ncu-rep)
  patterns/
    <strategy-family>_<hardware-id>.md    # one per (strategy, hardware) pair — YAML+Markdown
  lessons/
    <YYYYMMDD>_<slug>.md              # cross-cutting synthesized insights — YAML+Markdown
```

#### runs/ format — JSON

Key fields (full schema in `teco/knowledge/schema.py: RunRecord`):
```json
{
  "id": "run_20260221_143022_flash_attention_v2",
  "hardware": {"device_id": "a100-sxm4-80gb", "peak_tflops_fp16": 312.0, ...},
  "language": "triton",
  "kernel_characteristics": { ... },
  "shape_sweep": [{"label": "small", "dims": {"M": 64, "N": 64}}, ...],
  "baseline": {"profiling_by_shape": {"small": {"achieved_tflops": 12.1, "bottleneck": "latency"}, ...}},
  "strategies": [
    {
      "id": "s1", "name": "tensor-core-rewrite", "family": "tensor-core-rewrite",
      "status": "regime_winner",
      "winning_shape_range": {"description": "seq_q >= 512", "conditions": {"seq_q": ">= 512"}},
      "shape_results": {"small": {"achieved_tflops": 14.3}, "medium": {"achieved_tflops": 118.7}},
      "final_ncu": {"tensor_core_utilization": 0.87, "achieved_occupancy": 0.71}
    },
    {
      "id": "s2", "name": "vectorized-loads", "family": "vectorized-loads",
      "status": "regime_winner",
      "winning_shape_range": {"description": "seq_q < 512", "conditions": {"seq_q": "< 512"}},
      "shape_results": {"small": {"achieved_tflops": 89.3}, "medium": {"achieved_tflops": 71.2}}
    },
    {
      "id": "s3", "name": "shared-mem-tiling", "status": "pruned",
      "prune_reason": "dominated in all regimes by iter 1",
      "shape_results": {"small": {"achieved_tflops": 11.9}, "medium": {"achieved_tflops": 43.1}}
    }
  ],
  "dispatch_logic": "if seq_q < 512:\n    return vectorized_loads_impl(...)\nelse:\n    return tensor_core_impl(...)",
  "outcome": {"overall_speedup": 2.63, "regime_speedups": {"small": 7.38, "medium": 2.63}},
  "narrative": "## Problem\n...\n\n## Key Finding\n..."
}
```

#### patterns/ format — YAML front-matter + Markdown body

One file per `(strategy_family, hardware_id)`. Machine-maintained YAML front-matter;
human-editable Markdown body (never overwritten by automated updates).

```yaml
---
strategy_family: "tensor-core-rewrite"
hardware_id: "a100-sxm4-80gb"
run_count: 4
confidence: 0.89
applies_when:
  kernel_characteristics:
    compute_structure_requires: {tensor_core_parallelizable: ">= 0.7"}
    memory_access_pattern_allows: {coalesced: ">= 0.5"}
    arithmetic_intensity_class: ["high", "medium"]
    has_tensor_contractions: true
  shape_conditions:
    winning_regimes:
      - {description: "inner contraction dim >= 256", conditions: {K: ">= 256"}, confidence: 0.92}
    losing_regimes:
      - {description: "very small shapes", conditions: {total_elements: "< 4096"}, confidence: 0.81}
effect:
  bottlenecks_addressed: ["low_tensor_core_utilization", "compute_underutilization"]
  typical_speedup_range: [1.8, 4.2]
observations:
  - {run_id: "run_20260221_143022_flash_attention_v2", kernel_tags: ["attention", "reduction"],
     winning_regime: "seq_q >= 512", speedup_in_regime: 2.63,
     note: "Required replacing scalar loops with tl.dot()"}
  - {run_id: "run_20260119_091234_matmul_batched", kernel_tags: ["matmul"],
     winning_regime: "M >= 64 AND K >= 128", speedup_in_regime: 3.91}
implementation_notes:
  key_triton_primitives: ["tl.dot", "tl.constexpr for block sizes"]
  pitfalls:
    - "K must be multiple of 16 (fp16); add padding if needed"
    - "Avoid mixing tensor core and scalar paths — causes warp divergence"
---

## Strategy: Tensor Core Rewrite on A100

[Human-written narrative here — never auto-overwritten]
```

#### lessons/ format — YAML front-matter + Markdown body

Written by ReflectorAgent when a pattern repeats across >= 3 independent runs.

```yaml
---
title: "Tensor core strategies underperform at small shapes on A100"
hardware_ids: ["a100-sxm4-80gb"]
strategy_families_implicated: ["tensor-core-rewrite"]
confidence: 0.88
evidence_run_ids: ["run_20260221_...", "run_20260119_...", "run_20260115_..."]
triggers_when:
  bottleneck_from_ncu: ["latency_bound", "low_wave_count"]
  strategy_being_considered: ["tensor-core-rewrite"]
  shape_regime_label: ["small"]
recommendation: "prefer_alternative"
alternative_strategies: ["vectorized-loads"]
---

[Human-readable explanation of root cause and evidence]
```

#### Kernel characteristics schema

Structural properties of a kernel, extracted from code + ncu output. Used as the
primary similarity key for querying past patterns. Similarity to stored patterns is
structural ("random memory accesses", "tensor-core-parallelizable workload"), not
name/task based.

```python
class MemoryAccessWeights(BaseModel):
    """Weights in [0,1] representing prevalence; need not sum to 1."""
    coalesced: float = 0.0
    strided: float = 0.0
    random: float = 0.0
    gather_scatter: float = 0.0

class ComputeStructureWeights(BaseModel):
    tensor_core_parallelizable: float = 0.0
    vector_parallelizable: float = 0.0
    reduction: float = 0.0
    scan: float = 0.0
    irregular: float = 0.0

class KernelCharacteristics(BaseModel):
    memory_access_pattern: MemoryAccessWeights
    compute_structure: ComputeStructureWeights
    arithmetic_intensity_class: Literal["low", "medium", "high"]
    arithmetic_intensity_flop_per_byte: float | None = None
    data_types: list[str]
    dominant_data_type: str
    has_shared_memory_reuse: bool
    has_tensor_contractions: bool
    has_irregular_control_flow: bool = False
    parallelism_axes: list[str]
    reduction_axes: list[str]
    characteristic_tags: list[str]     # free-form, e.g. ["attention", "softmax"]
    extraction_confidence: float = 0.7
```

Weighted fields allow fuzzy representation: a kernel that is "mostly coalesced with some
scatter" is encoded as `{coalesced: 0.8, gather_scatter: 0.2}` and partially matches
patterns requiring coalesced >= 0.5.

#### KnowledgeStore query interface

```python
class KnowledgeStore:
    def query(
        self,
        characteristics: KernelCharacteristics,
        hardware_id: str,
        shape_point: ShapePoint | None = None,
        bottleneck_hints: list[str] | None = None,
    ) -> QueryResult:
        """Main entry point. Runs all query types and returns merged ranked results."""

    def upsert_pattern(
        self,
        strategy_family: str,
        hardware_id: str,
        new_observation: PatternObservation,
        run_record: RunRecord,
    ) -> Pattern:
        """Incremental update: appends observation, re-derives aggregates, preserves Markdown body."""

    def write_run(self, record: RunRecord) -> Path: ...
    def write_lesson(self, lesson: Lesson) -> Path: ...
    def rebuild_index(self) -> None: ...
```

**Matching strategy (no vector DB) — three-pass funnel:**

1. **Hard filter** via `index.json`: exact match on `hardware_id`, optionally
   `arithmetic_intensity_class` and `data_types`. O(1), avoids file I/O.
2. **Composite score** on remaining candidates:
   ```
   score = 0.40 * characteristics_score   # weighted-vector threshold matching
         + 0.25 * shape_score             # partial credit on numeric conditions
         + 0.25 * bottleneck_score        # Jaccard overlap on bottleneck labels
         + 0.10 * tag_overlap_score       # Jaccard on characteristic_tags
   ```
   Shape partial credit: condition `{K: ">= 256"}` against `K=128` scores `128/256 = 0.5`.
3. **Rank and return** top-k, filtered by `pattern.confidence`.

Lessons use a disjunctive trigger: surface if ANY of `bottleneck_from_ncu`,
`strategy_being_considered`, or `shape_regime_label` intersects with the query.

#### index.json structure

Lightweight metadata cache — avoids full directory scans. Auto-rebuilt via
`store.rebuild_index()` after manual edits.

```json
{
  "runs": {
    "run_20260221_143022_flash_attention_v2": {
      "path": "knowledge/runs/run_20260221_143022_flash_attention_v2.json",
      "hardware_id": "a100-sxm4-80gb", "language": "triton",
      "characteristic_tags": ["attention", "softmax"],
      "arithmetic_intensity_class": "high",
      "strategy_families_tested": ["tensor-core-rewrite", "vectorized-loads"]
    }
  },
  "patterns": {
    "pat_tensor-core-rewrite_a100-sxm4-80gb": {
      "path": "knowledge/patterns/tensor-core-rewrite_a100-sxm4-80gb.md",
      "strategy_family": "tensor-core-rewrite", "hardware_id": "a100-sxm4-80gb",
      "run_count": 4, "confidence": 0.89,
      "arithmetic_intensity_classes": ["high", "medium"],
      "bottlenecks_addressed": ["low_tensor_core_utilization"],
      "min_tensor_core_parallelizable": 0.7, "has_tensor_contractions": true
    }
  },
  "lessons": {
    "lesson_20260221_small-shape-tensor-core-penalty-a100": {
      "path": "knowledge/lessons/20260221_small-shape-tensor-core-penalty-a100.md",
      "triggers_on_strategies": ["tensor-core-rewrite"],
      "triggers_on_shape_labels": ["small"], "recommendation": "prefer_alternative"
    }
  }
}
```

Index written atomically (write to `.index.json.tmp`, then rename) to prevent corruption.

### TritonBench integration (for testing TECO itself)
- Input: `TritonBench/data/TritonBench_G_v1.json` entries
- Use `scratch.py` pattern to load entries
- Evaluation: compare against `performance_metrics/perf_G/golden_results/`

---

## Dependencies (pyproject.toml)

```toml
[project]
dependencies = [
  "smolagents",
  "anthropic",
  "triton",
  "torch",
  "pynvml",       # GPU hardware ceiling queries
  "pydantic>=2",
  "typer",
  "httpx",        # for doc fetching (Learner, later)
  "ruamel.yaml",  # round-trip YAML parsing (preserves hand-edited pattern/lesson files)
]

[tool.uv]  # package manager to confirm
# System requirement: ncu (Nsight Compute) must be on PATH
```

---

## Files to create (in order)

1. `pyproject.toml` — project metadata + dependencies
2. `teco/knowledge/schema.py` — Pydantic models
3. `teco/knowledge/store.py` — KnowledgeStore (read/write Markdown+JSON)
4. `teco/tools/profiler.py` — two-stage GPU profiling tool
5. `teco/tools/compiler.py` — compile + validate tool
6. `teco/tools/code_editor.py` — apply patch tool
7. `teco/agents/base.py` — BaseAgent
8. `teco/agents/optimizer.py` — OptimizerAgent (full)
9. `teco/agents/learner.py`, `experimenter.py`, `reflector.py` — stubs
10. `teco/orchestration/context.py` + `loop.py`
11. `teco/cli.py` — `teco optimize <kernel.py> --hardware a100 --iterations 10`
12. `tests/` — unit + integration tests
13. Update `CLAUDE.md` with finalized toolchain

---

## Verification

1. `uv run python scratch.py` — loads TritonBench, confirms data format
2. `uv run teco optimize TritonBench/data/TritonBench_G_v1/lightning_attention.py` — runs optimizer loop end-to-end
3. `uv run pytest tests/` — unit tests pass
4. Check `knowledge/runs/` for generated run report
5. Compare final TFLOPS against TritonBench golden results

---

## Future extension points (not in MVP)

- LearnerAgent: crawl Triton docs, arXiv, GitHub issues before optimization
- ExperimenterAgent: spawn isolated microbenchmarks (block size sweep, occupancy test)
  when bottleneck is ambiguous; self-reflect to generalize findings
- ReflectorAgent: LLM-powered pattern extraction from run logs
- Multi-language: pluggable compiler backend (CUDA via nvcc, TileLang, AscendC)
- Remote execution: abstract `ExecutionBackend` (local / SSH / SLURM)
- Vector similarity search over knowledge/ using embeddings
- Translation mode: rewrite kernel from one language to another (e.g., Triton → AscendC)
