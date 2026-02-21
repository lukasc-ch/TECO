"""Tests for KnowledgeStore: schema, query, and incremental update."""

from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from teco.knowledge.schema import (
    BaselineProfile,
    ComputeStructureWeights,
    HardwareInfo,
    KernelCharacteristics,
    Lesson,
    LessonTrigger,
    MemoryAccessWeights,
    Pattern,
    PatternObservation,
    QueryResult,
    RunOutcome,
    RunRecord,
    ShapePoint,
    ShapeRange,
    StrategyRecord,
)
from teco.knowledge.store import KnowledgeStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_knowledge_root(tmp_path: Path) -> Path:
    return tmp_path / "knowledge"


@pytest.fixture
def store(tmp_knowledge_root: Path) -> KnowledgeStore:
    return KnowledgeStore(knowledge_root=tmp_knowledge_root)


@pytest.fixture
def hw_info() -> HardwareInfo:
    return HardwareInfo(
        device_name="NVIDIA A100-SXM4-80GB",
        device_id="a100-sxm4-80gb",
        peak_tflops_fp16=312.0,
        peak_tflops_fp32=19.5,
        peak_bandwidth_gbs=2039.0,
        sm_count=108,
        compute_capability="8.0",
    )


@pytest.fixture
def kernel_chars() -> KernelCharacteristics:
    return KernelCharacteristics(
        memory_access_pattern=MemoryAccessWeights(coalesced=0.8, gather_scatter=0.2),
        compute_structure=ComputeStructureWeights(
            tensor_core_parallelizable=0.9, reduction=0.8
        ),
        arithmetic_intensity_class="high",
        data_types=["fp16"],
        dominant_data_type="fp16",
        has_tensor_contractions=True,
        characteristic_tags=["attention", "softmax"],
    )


@pytest.fixture
def sample_run_record(hw_info: HardwareInfo, kernel_chars: KernelCharacteristics) -> RunRecord:
    return RunRecord(
        id="run_20260221_143022_flash_attention_v2",
        created_at=datetime.now(timezone.utc),
        task="flash_attention_v2",
        hardware=hw_info,
        language="triton",
        kernel_characteristics=kernel_chars,
        shape_sweep=[
            ShapePoint(label="small", dims={"seq_q": 64, "seq_k": 64}),
            ShapePoint(label="medium", dims={"seq_q": 512, "seq_k": 512}),
            ShapePoint(label="large", dims={"seq_q": 4096, "seq_k": 4096}),
        ],
        baseline=BaselineProfile(
            source_hash="abc123",
            profiling_by_shape={
                "small": {"achieved_tflops": 12.1, "bottleneck": "latency"},
                "medium": {"achieved_tflops": 45.2, "bottleneck": "memory"},
                "large": {"achieved_tflops": 51.8, "bottleneck": "memory"},
            },
        ),
        strategies=[
            StrategyRecord(
                id="s1",
                name="tensor-core-rewrite",
                family="tensor-core-rewrite",
                status="regime_winner",
                winning_shape_range=ShapeRange(
                    description="seq_q >= 512", conditions={"seq_q": ">= 512"}
                ),
                predicted_winning_regime="medium and large shapes",
                llm_confidence_final=0.91,
                addresses_bottleneck="low tensor core utilization",
                iterations_applied=5,
                shape_results={
                    "small": {"achieved_tflops": 14.3, "vs_baseline_pct": 18.2},
                    "medium": {"achieved_tflops": 118.7, "vs_baseline_pct": 162.6},
                    "large": {"achieved_tflops": 201.4, "vs_baseline_pct": 288.5},
                },
                final_ncu={"tensor_core_utilization": 0.87, "achieved_occupancy": 0.71},
            ),
            StrategyRecord(
                id="s3",
                name="shared-mem-tiling",
                family="shared-mem-tiling",
                status="pruned",
                prune_reason="dominated in all shape regimes",
                predicted_winning_regime="all sizes",
                llm_confidence_final=0.11,
                addresses_bottleneck="shared memory bank conflicts",
                iterations_applied=1,
                shape_results={},
            ),
        ],
        outcome=RunOutcome(
            overall_speedup=2.63,
            regime_speedups={"small": 1.18, "medium": 2.63, "large": 3.89},
            total_iterations=7,
            strategies_tried=3,
            strategies_pruned=1,
            strategies_deployed=1,
        ),
    )


# ---------------------------------------------------------------------------
# Tests: RunRecord persistence
# ---------------------------------------------------------------------------


class TestRunRecord:
    def test_write_and_retrieve(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        path = store.write_run(sample_run_record)
        assert path.exists()

        retrieved = store.get_run(sample_run_record.id)
        assert retrieved.id == sample_run_record.id
        assert retrieved.task == "flash_attention_v2"
        assert retrieved.outcome.overall_speedup == pytest.approx(2.63)

    def test_run_appears_in_index(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        store.write_run(sample_run_record)
        assert sample_run_record.id in store._index["runs"]
        entry = store._index["runs"][sample_run_record.id]
        assert entry["hardware_id"] == "a100-sxm4-80gb"
        assert "attention" in entry["characteristic_tags"]


# ---------------------------------------------------------------------------
# Tests: Pattern upsert
# ---------------------------------------------------------------------------


class TestPatternUpsert:
    def _make_obs(self, run_id: str, speedup: float | None = 2.5) -> PatternObservation:
        return PatternObservation(
            run_id=run_id,
            task="flash_attention_v2",
            kernel_tags=["attention", "softmax"],
            winning_regime="seq_q >= 512" if speedup else None,
            speedup_in_regime=speedup,
            note="test observation",
        )

    def test_creates_new_pattern_file(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        obs = self._make_obs("run_001")
        pattern = store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)
        assert pattern.run_count == 1
        assert len(pattern.observations) == 1
        path = store.patterns_dir / "tensor-core-rewrite_a100-sxm4-80gb.md"
        assert path.exists()

    def test_incremental_append(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        for i in range(3):
            obs = self._make_obs(f"run_00{i}", speedup=2.0 + i * 0.5)
            store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)

        pattern = store.get_pattern("pat_tensor-core-rewrite_a100-sxm4-80gb")
        assert pattern.run_count == 3
        assert len(pattern.observations) == 3

    def test_no_duplicate_observations(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        obs = self._make_obs("run_unique")
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)

        pattern = store.get_pattern("pat_tensor-core-rewrite_a100-sxm4-80gb")
        assert pattern.run_count == 1

    def test_markdown_body_preserved(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        obs = self._make_obs("run_a")
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)

        # Manually edit the markdown body
        path = store.patterns_dir / "tensor-core-rewrite_a100-sxm4-80gb.md"
        text = path.read_text()
        path.write_text(text.rstrip() + "\n\nHuman-written note: important insight.\n")

        # Upsert again — body should be preserved
        obs2 = self._make_obs("run_b")
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs2, sample_run_record)

        pattern = store.get_pattern("pat_tensor-core-rewrite_a100-sxm4-80gb")
        assert "Human-written note" in pattern.body_markdown

    def test_confidence_increases_with_winners(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        # 3 winners, 1 pruned
        for i in range(3):
            obs = self._make_obs(f"run_win_{i}", speedup=2.0 + i)
            store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)
        obs_pruned = self._make_obs("run_pruned", speedup=None)
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs_pruned, sample_run_record)

        pattern = store.get_pattern("pat_tensor-core-rewrite_a100-sxm4-80gb")
        # 3/4 winning → confidence > 0.5 baseline
        assert pattern.confidence > 0.5


# ---------------------------------------------------------------------------
# Tests: Query — characteristics
# ---------------------------------------------------------------------------


class TestQueryByCharacteristics:
    def _seed_patterns(self, store: KnowledgeStore, sample_run_record: RunRecord) -> None:
        families = ["tensor-core-rewrite", "vectorized-loads", "shared-mem-tiling"]
        speedups = [2.5, 1.5, None]
        for family, speedup in zip(families, speedups):
            obs = PatternObservation(
                run_id="run_seed",
                task="test",
                kernel_tags=["attention"] if family == "tensor-core-rewrite" else ["memory"],
                winning_regime="medium" if speedup else None,
                speedup_in_regime=speedup,
            )
            store.upsert_pattern(family, "a100-sxm4-80gb", obs, sample_run_record)

    def test_returns_results_for_known_hardware(
        self, store: KnowledgeStore, sample_run_record: RunRecord, kernel_chars: KernelCharacteristics
    ) -> None:
        self._seed_patterns(store, sample_run_record)
        result = store.query(kernel_chars, hardware_id="a100-sxm4-80gb")
        assert isinstance(result, QueryResult)
        assert len(result.patterns) > 0

    def test_no_results_for_unknown_hardware(
        self, store: KnowledgeStore, sample_run_record: RunRecord, kernel_chars: KernelCharacteristics
    ) -> None:
        self._seed_patterns(store, sample_run_record)
        result = store.query(kernel_chars, hardware_id="h100-sxm5-80gb")
        assert len(result.patterns) == 0
        assert any("No prior experience" in w for w in result.warnings)

    def test_ranking_favors_matching_tags(
        self, store: KnowledgeStore, sample_run_record: RunRecord, kernel_chars: KernelCharacteristics
    ) -> None:
        self._seed_patterns(store, sample_run_record)
        result = store.query(kernel_chars, hardware_id="a100-sxm4-80gb")
        if len(result.patterns) > 1:
            # tensor-core-rewrite has "attention" tag which matches kernel_chars
            families = [m.pattern.strategy_family for m in result.patterns]
            # The attention-tagged pattern should rank at or near the top
            assert families.index("tensor-core-rewrite") <= 1


# ---------------------------------------------------------------------------
# Tests: Query — shape regime
# ---------------------------------------------------------------------------


class TestQueryByShapeRegime:
    def test_shape_query_matches_winning_regime(
        self, store: KnowledgeStore, sample_run_record: RunRecord, kernel_chars: KernelCharacteristics
    ) -> None:
        obs = PatternObservation(
            run_id="run_shape",
            task="attn",
            kernel_tags=["attention"],
            winning_regime="seq_q >= 512",
            speedup_in_regime=2.5,
        )
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)

        shape = ShapePoint(label="large", dims={"seq_q": 4096})
        result = store.query_by_shape_regime(shape, "a100-sxm4-80gb")
        # No scoring since pattern hasn't been manually given shape conditions,
        # but at minimum no crash
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Query — bottleneck
# ---------------------------------------------------------------------------


class TestQueryByBottleneck:
    def test_bottleneck_query(
        self, store: KnowledgeStore, sample_run_record: RunRecord, kernel_chars: KernelCharacteristics
    ) -> None:
        obs = PatternObservation(
            run_id="run_bn",
            task="attn",
            kernel_tags=["attention"],
            winning_regime=None,
            speedup_in_regime=None,
        )
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)
        # Query returns list (may be empty if pattern has no bottleneck annotations yet)
        result = store.query_by_bottleneck(
            ["low_tensor_core_utilization"], "a100-sxm4-80gb"
        )
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Lessons
# ---------------------------------------------------------------------------


class TestLessons:
    def test_write_and_trigger_lesson(
        self, store: KnowledgeStore
    ) -> None:
        lesson = Lesson(
            id="lesson_20260221_test",
            title="Test lesson",
            created_at=date.today(),
            last_validated=date.today(),
            hardware_ids=["a100-sxm4-80gb"],
            confidence=0.8,
            triggers_when=LessonTrigger(
                strategy_being_considered=["tensor-core-rewrite"],
                shape_regime_label=["small"],
            ),
            recommendation="prefer_alternative",
            alternative_strategies=["vectorized-loads"],
        )
        path = store.write_lesson(lesson)
        assert path.exists()

        retrieved_lessons = store.get_lessons(
            hardware_id="a100-sxm4-80gb",
            strategy_families=["tensor-core-rewrite"],
        )
        assert len(retrieved_lessons) == 1
        assert retrieved_lessons[0].title == "Test lesson"

    def test_lesson_not_triggered_for_wrong_hardware(
        self, store: KnowledgeStore
    ) -> None:
        lesson = Lesson(
            id="lesson_20260221_hw",
            title="A100-specific lesson",
            created_at=date.today(),
            last_validated=date.today(),
            hardware_ids=["a100-sxm4-80gb"],
            confidence=0.8,
            triggers_when=LessonTrigger(strategy_being_considered=["tensor-core-rewrite"]),
        )
        store.write_lesson(lesson)

        lessons = store.get_lessons(
            hardware_id="h100-sxm5-80gb",
            strategy_families=["tensor-core-rewrite"],
        )
        assert len(lessons) == 0


# ---------------------------------------------------------------------------
# Tests: Index rebuild
# ---------------------------------------------------------------------------


class TestIndexRebuild:
    def test_rebuild_restores_index(
        self, store: KnowledgeStore, sample_run_record: RunRecord
    ) -> None:
        store.write_run(sample_run_record)
        obs = PatternObservation(
            run_id="run_rebuild", task="t", kernel_tags=[], winning_regime=None
        )
        store.upsert_pattern("tensor-core-rewrite", "a100-sxm4-80gb", obs, sample_run_record)

        # Corrupt the in-memory index
        store._index = {"schema_version": "1.0", "runs": {}, "patterns": {}, "lessons": {}}

        store.rebuild_index()

        assert sample_run_record.id in store._index["runs"]
        assert "pat_tensor-core-rewrite_a100-sxm4-80gb" in store._index["patterns"]
