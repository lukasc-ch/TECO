"""Pydantic models for the TECO knowledge store."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


class HardwareInfo(BaseModel):
    device_name: str
    device_id: str  # normalized slug, e.g. "a100-sxm4-80gb"
    peak_tflops_fp16: float
    peak_tflops_fp32: float
    peak_bandwidth_gbs: float
    sm_count: int
    compute_capability: str  # e.g. "8.0"


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


class ProfilingReport(BaseModel):
    """Rich ncu-derived profiling metrics for a single (kernel, shape) pair."""

    # Roofline
    achieved_tflops: float
    peak_tflops: float  # from pynvml for the active dtype
    arithmetic_intensity: float  # FLOP/byte
    bottleneck: Literal["memory", "compute", "latency"]
    utilization_pct: float  # max(achieved/peak_tflops, achieved_bw/peak_bw) * 100

    # Memory hierarchy
    l1_hit_rate: float
    l2_hit_rate: float
    dram_bandwidth_gbs: float
    peak_bandwidth_gbs: float  # from pynvml
    shared_mem_bank_conflicts: int
    global_load_efficiency: float
    global_store_efficiency: float

    # Warp / occupancy
    warp_efficiency: float
    branch_divergence_pct: float
    achieved_occupancy: float
    theoretical_occupancy: float
    instruction_replay_overhead: float

    # Instruction mix
    tensor_core_utilization: float
    fp16_ops: int
    fp32_ops: int
    int_ops: int
    special_func_ops: int

    # Latency breakdown (Stage 1)
    latency_ms: float

    # Raw report path for agent drill-down
    raw_ncu_report_path: str = ""


# ---------------------------------------------------------------------------
# Shape
# ---------------------------------------------------------------------------


class ShapePoint(BaseModel):
    label: str  # "small" | "medium" | "large" | custom
    dims: dict[str, int]  # e.g. {"M": 512, "N": 1024, "K": 64}


class ShapeRange(BaseModel):
    description: str  # human-readable, e.g. "seq_q >= 512"
    conditions: dict[str, str]  # parseable, e.g. {"seq_q": ">= 512"}


# ---------------------------------------------------------------------------
# Kernel characteristics  (structural similarity key)
# ---------------------------------------------------------------------------


class MemoryAccessWeights(BaseModel):
    """Prevalence weights in [0, 1]; need not sum to 1."""

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
    memory_access_pattern: MemoryAccessWeights = Field(default_factory=MemoryAccessWeights)
    compute_structure: ComputeStructureWeights = Field(default_factory=ComputeStructureWeights)

    arithmetic_intensity_class: Literal["low", "medium", "high"] = "medium"
    arithmetic_intensity_flop_per_byte: float | None = None

    data_types: list[str] = Field(default_factory=list)  # ["fp16", "fp32", ...]
    dominant_data_type: str = "fp32"

    # Boolean structural properties
    has_shared_memory_reuse: bool = False
    has_tensor_contractions: bool = False
    has_online_softmax: bool = False
    has_causal_mask: bool = False
    has_irregular_control_flow: bool = False

    # Parallelism topology
    parallelism_axes: list[str] = Field(default_factory=list)  # e.g. ["batch", "heads"]
    reduction_axes: list[str] = Field(default_factory=list)

    # Free-form tags for substring/Jaccard matching
    characteristic_tags: list[str] = Field(default_factory=list)

    # Agent's confidence in its own analysis
    extraction_confidence: float = 0.7


# ---------------------------------------------------------------------------
# Strategy tree (maintained during an optimization run)
# ---------------------------------------------------------------------------


class StrategyShapeResult(BaseModel):
    shape_point: ShapePoint
    achieved_tflops: float
    vs_baseline_pct: float  # percentage change vs baseline at same shape
    profiling_report: ProfilingReport | None = None


class Strategy(BaseModel):
    id: str
    name: str  # e.g. "tensor-core-rewrite"
    family: str  # normalized slug for pattern lookup
    rationale: str
    addresses_bottleneck: str
    implementation_sketch: str  # rough description, not full code
    predicted_winning_regime: str  # LLM free-text prediction
    incompatible_with: list[str] = Field(default_factory=list)  # other strategy ids

    llm_confidence: float = 0.5  # updated each iteration
    status: Literal["active", "pruned", "regime_winner"] = "active"
    winning_shape_range: ShapeRange | None = None
    prune_reason: str | None = None

    shape_results: list[StrategyShapeResult] = Field(default_factory=list)
    current_code: str = ""  # latest implementation
    iterations_applied: int = 0


class StrategyTree(BaseModel):
    strategies: list[Strategy] = Field(default_factory=list)
    shape_sweep: list[ShapePoint] = Field(default_factory=list)
    active_ids: list[str] = Field(default_factory=list)
    regime_winners: dict[str, str] = Field(default_factory=dict)  # strategy_id → regime desc
    iteration: int = 0


# ---------------------------------------------------------------------------
# Run record  (ground-truth log, written to runs/)
# ---------------------------------------------------------------------------


class BaselineProfile(BaseModel):
    source_hash: str  # sha256 of original kernel file
    profiling_by_shape: dict[str, dict[str, Any]]  # label → {achieved_tflops, bottleneck, ...}


class RunOutcome(BaseModel):
    overall_speedup: float
    regime_speedups: dict[str, float]  # shape label → speedup
    total_iterations: int
    strategies_tried: int
    strategies_pruned: int
    strategies_deployed: int  # regime winners in dispatch wrapper


class StrategyRecord(BaseModel):
    """Serialized strategy for storage in RunRecord (simplified vs in-memory Strategy)."""

    id: str
    name: str
    family: str
    status: Literal["active", "pruned", "regime_winner"]
    winning_shape_range: ShapeRange | None = None
    prune_reason: str | None = None
    predicted_winning_regime: str
    llm_confidence_final: float
    addresses_bottleneck: str
    incompatible_with: list[str] = Field(default_factory=list)
    iterations_applied: int
    shape_results: dict[str, dict[str, float]] = Field(default_factory=dict)  # label → metrics
    final_ncu: dict[str, float] | None = None


class RunRecord(BaseModel):
    schema_version: str = "1.0"
    id: str
    created_at: datetime
    task: str
    hardware: HardwareInfo
    language: str
    kernel_characteristics: KernelCharacteristics
    shape_sweep: list[ShapePoint]
    baseline: BaselineProfile
    strategies: list[StrategyRecord]
    dispatch_logic: str | None = None
    outcome: RunOutcome
    ncu_report_paths: dict[str, str] = Field(default_factory=dict)
    narrative: str = ""  # free-form Markdown summary


# ---------------------------------------------------------------------------
# Pattern  (per strategy_family × hardware, stored in patterns/)
# ---------------------------------------------------------------------------


class ShapeCondition(BaseModel):
    description: str
    conditions: dict[str, str]
    confidence: float


class PatternObservation(BaseModel):
    run_id: str
    task: str
    kernel_tags: list[str]
    winning_regime: str | None = None
    speedup_in_regime: float | None = None
    note: str = ""


class PatternAppliesWhen(BaseModel):
    kernel_characteristics: dict[str, Any] = Field(default_factory=dict)
    shape_conditions: dict[str, list[ShapeCondition]] = Field(
        default_factory=lambda: {"winning_regimes": [], "losing_regimes": []}
    )


class PatternEffect(BaseModel):
    bottlenecks_addressed: list[str] = Field(default_factory=list)
    bottlenecks_not_addressed: list[str] = Field(default_factory=list)
    typical_speedup_range: tuple[float, float] = (1.0, 1.0)
    speedup_variance: str = "unknown"


class PatternImplementationNotes(BaseModel):
    key_triton_primitives: list[str] = Field(default_factory=list)
    recommended_block_sizes: list[dict[str, Any]] = Field(default_factory=list)
    pitfalls: list[str] = Field(default_factory=list)
    incompatible_strategies: list[str] = Field(default_factory=list)


class Pattern(BaseModel):
    schema_version: str = "1.0"
    id: str  # e.g. "pat_tensor-core-rewrite_a100-sxm4-80gb"
    strategy_family: str
    hardware_id: str
    last_updated: date
    run_count: int = 0
    source_run_ids: list[str] = Field(default_factory=list)
    applies_when: PatternAppliesWhen = Field(default_factory=PatternAppliesWhen)
    effect: PatternEffect = Field(default_factory=PatternEffect)
    observations: list[PatternObservation] = Field(default_factory=list)
    implementation_notes: PatternImplementationNotes = Field(
        default_factory=PatternImplementationNotes
    )
    confidence: float = 0.5
    body_markdown: str = ""  # human-editable; never auto-overwritten


# ---------------------------------------------------------------------------
# Lesson  (cross-cutting insights, stored in lessons/)
# ---------------------------------------------------------------------------


class LessonPredicate(BaseModel):
    hardware_ids: list[str] = Field(default_factory=list)
    hardware_families: list[str] = Field(default_factory=list)
    kernel_characteristics: dict[str, Any] = Field(default_factory=dict)
    shape_predicates: list[dict[str, Any]] = Field(default_factory=list)


class LessonTrigger(BaseModel):
    bottleneck_from_ncu: list[str] = Field(default_factory=list)
    strategy_being_considered: list[str] = Field(default_factory=list)
    shape_regime_label: list[str] = Field(default_factory=list)


class Lesson(BaseModel):
    schema_version: str = "1.0"
    id: str
    title: str
    created_at: date
    last_validated: date
    hardware_ids: list[str] = Field(default_factory=list)
    hardware_families: list[str] = Field(default_factory=list)
    strategy_families_implicated: list[str] = Field(default_factory=list)
    contradicts_lesson_ids: list[str] = Field(default_factory=list)
    supersedes_lesson_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.5
    evidence_run_ids: list[str] = Field(default_factory=list)
    applies_to: LessonPredicate = Field(default_factory=LessonPredicate)
    triggers_when: LessonTrigger = Field(default_factory=LessonTrigger)
    recommendation: Literal[
        "prefer_alternative", "apply_with_caution", "avoid", "always_try"
    ] = "apply_with_caution"
    alternative_strategies: list[str] = Field(default_factory=list)
    body_markdown: str = ""


# ---------------------------------------------------------------------------
# Query results
# ---------------------------------------------------------------------------


class PatternMatch(BaseModel):
    pattern: Pattern
    score: float  # 0–1 composite
    score_breakdown: dict[str, float]  # {"characteristics": 0.9, "shape": 0.7, ...}
    matched_observations: list[PatternObservation] = Field(default_factory=list)


class QueryResult(BaseModel):
    patterns: list[PatternMatch] = Field(default_factory=list)  # ranked by score desc
    lessons: list[Lesson] = Field(default_factory=list)
    suggested_strategy_families: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
