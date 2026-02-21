"""KnowledgeStore: read/write/query the hierarchical knowledge base."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from .schema import (
    KernelCharacteristics,
    Lesson,
    LessonTrigger,
    Pattern,
    PatternAppliesWhen,
    PatternEffect,
    PatternImplementationNotes,
    PatternMatch,
    PatternObservation,
    QueryResult,
    RunRecord,
    ShapePoint,
)

yaml = YAML()
yaml.default_flow_style = False
yaml.preserve_quotes = True

# ---------------------------------------------------------------------------
# Index entry types (lightweight, kept in memory + index.json)
# ---------------------------------------------------------------------------

RunIndexEntry = dict[str, Any]
PatternIndexEntry = dict[str, Any]
LessonIndexEntry = dict[str, Any]

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# KnowledgeStore
# ---------------------------------------------------------------------------


class KnowledgeStore:
    def __init__(self, knowledge_root: Path | str = "knowledge") -> None:
        self.root = Path(knowledge_root)
        self.runs_dir = self.root / "runs"
        self.patterns_dir = self.root / "patterns"
        self.lessons_dir = self.root / "lessons"
        self.index_path = self.root / "index.json"

        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.patterns_dir.mkdir(parents=True, exist_ok=True)
        self.lessons_dir.mkdir(parents=True, exist_ok=True)

        self._index: dict[str, Any] = self._load_index()

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            with self.index_path.open() as f:
                return json.load(f)
        return {
            "schema_version": SCHEMA_VERSION,
            "last_updated": "",
            "runs": {},
            "patterns": {},
            "lessons": {},
        }

    def _save_index(self) -> None:
        self._index["last_updated"] = datetime.now(timezone.utc).isoformat()
        tmp = self.index_path.with_suffix(".json.tmp")
        with tmp.open("w") as f:
            json.dump(self._index, f, indent=2)
        os.replace(tmp, self.index_path)

    def rebuild_index(self) -> None:
        """Full re-scan of all files. Use after manual edits."""
        self._index = {
            "schema_version": SCHEMA_VERSION,
            "last_updated": "",
            "runs": {},
            "patterns": {},
            "lessons": {},
        }
        for path in self.runs_dir.glob("*.json"):
            with path.open() as f:
                data = json.load(f)
            self._index["runs"][data["id"]] = self._run_index_entry(data, path)

        for path in self.patterns_dir.glob("*.md"):
            front, _ = _split_yaml_markdown(path.read_text())
            self._index["patterns"][front["id"]] = self._pattern_index_entry(front, path)

        for path in self.lessons_dir.glob("*.md"):
            front, _ = _split_yaml_markdown(path.read_text())
            self._index["lessons"][front["id"]] = self._lesson_index_entry(front, path)

        self._save_index()

    @staticmethod
    def _run_index_entry(data: dict[str, Any], path: Path) -> RunIndexEntry:
        chars = data.get("kernel_characteristics", {})
        return {
            "path": str(path),
            "created_at": data.get("created_at", ""),
            "task": data.get("task", ""),
            "hardware_id": data.get("hardware", {}).get("device_id", ""),
            "language": data.get("language", ""),
            "characteristic_tags": chars.get("characteristic_tags", []),
            "arithmetic_intensity_class": chars.get("arithmetic_intensity_class", ""),
            "dominant_data_type": chars.get("dominant_data_type", ""),
            "has_tensor_contractions": chars.get("has_tensor_contractions", False),
            "strategy_families_tested": [
                s.get("family", "") for s in data.get("strategies", [])
            ],
            "outcome_speedup": data.get("outcome", {}).get("overall_speedup", 0.0),
        }

    @staticmethod
    def _pattern_index_entry(front: dict[str, Any], path: Path) -> PatternIndexEntry:
        aw = front.get("applies_when", {})
        kc = aw.get("kernel_characteristics", {})
        eff = front.get("effect", {})
        return {
            "path": str(path),
            "strategy_family": front.get("strategy_family", ""),
            "hardware_id": front.get("hardware_id", ""),
            "last_updated": str(front.get("last_updated", "")),
            "run_count": front.get("run_count", 0),
            "confidence": front.get("confidence", 0.0),
            "arithmetic_intensity_classes": kc.get("arithmetic_intensity_class", []),
            "data_types": kc.get("data_types_any_of", []),
            "has_tensor_contractions": kc.get("has_tensor_contractions", False),
            "min_tensor_core_parallelizable": _parse_threshold(
                kc.get("compute_structure_requires", {}).get("tensor_core_parallelizable", "0")
            ),
            "min_coalesced": _parse_threshold(
                kc.get("memory_access_pattern_allows", {}).get("coalesced", "0")
            ),
            "bottlenecks_addressed": eff.get("bottlenecks_addressed", []),
            "typical_speedup_min": (eff.get("typical_speedup_range") or [1.0, 1.0])[0],
            "typical_speedup_max": (eff.get("typical_speedup_range") or [1.0, 1.0])[-1],
            "winning_regimes_summary": [
                c.get("description", "")
                for c in aw.get("shape_conditions", {}).get("winning_regimes", [])
            ],
        }

    @staticmethod
    def _lesson_index_entry(front: dict[str, Any], path: Path) -> LessonIndexEntry:
        tw = front.get("triggers_when", {})
        return {
            "path": str(path),
            "title": front.get("title", ""),
            "created_at": str(front.get("created_at", "")),
            "hardware_ids": front.get("hardware_ids", []),
            "hardware_families": front.get("hardware_families", []),
            "strategy_families_implicated": front.get("strategy_families_implicated", []),
            "confidence": front.get("confidence", 0.0),
            "triggers_on_bottlenecks": tw.get("bottleneck_from_ncu", []),
            "triggers_on_strategies": tw.get("strategy_being_considered", []),
            "triggers_on_shape_labels": tw.get("shape_regime_label", []),
            "recommendation": front.get("recommendation", "apply_with_caution"),
        }

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------

    def write_run(self, record: RunRecord) -> Path:
        path = self.runs_dir / f"{record.id}.json"
        with path.open("w") as f:
            json.dump(json.loads(record.model_dump_json()), f, indent=2, default=str)
        self._index["runs"][record.id] = self._run_index_entry(
            json.loads(record.model_dump_json()), path
        )
        self._save_index()
        return path

    def upsert_pattern(
        self,
        strategy_family: str,
        hardware_id: str,
        new_observation: PatternObservation,
        run_record: RunRecord,
    ) -> Pattern:
        """
        Append observation to existing pattern file, re-derive aggregates.
        Preserves the Markdown body on incremental updates.
        """
        slug = f"{strategy_family}_{hardware_id}"
        path = self.patterns_dir / f"{slug}.md"

        if path.exists():
            pattern = self._load_pattern(path)
        else:
            pattern = self._bootstrap_pattern(strategy_family, hardware_id, run_record)

        # Append new observation if not already present
        existing_ids = {o.run_id for o in pattern.observations}
        if new_observation.run_id not in existing_ids:
            pattern.observations.append(new_observation)
            pattern.run_count += 1
            if run_record.id not in pattern.source_run_ids:
                pattern.source_run_ids.append(run_record.id)

        pattern.last_updated = date.today()
        pattern = self._re_derive_aggregates(pattern)

        self._write_pattern(pattern, path)
        self._index["patterns"][pattern.id] = self._pattern_index_entry(
            self._pattern_to_front_matter(pattern), path
        )
        self._save_index()
        return pattern

    def write_lesson(self, lesson: Lesson) -> Path:
        slug = lesson.id.replace("lesson_", "").replace("_", "-", 1)
        path = self.lessons_dir / f"{slug}.md"
        self._write_lesson(lesson, path)
        self._index["lessons"][lesson.id] = self._lesson_index_entry(
            self._lesson_to_front_matter(lesson), path
        )
        self._save_index()
        return path

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------

    def query(
        self,
        characteristics: KernelCharacteristics,
        hardware_id: str,
        shape_point: ShapePoint | None = None,
        bottleneck_hints: list[str] | None = None,
        strategy_families: list[str] | None = None,
        max_results: int = 10,
    ) -> QueryResult:
        patterns = self.query_by_characteristics(characteristics, hardware_id, top_k=max_results)

        if shape_point is not None:
            shape_matches = self.query_by_shape_regime(shape_point, hardware_id)
            patterns = _merge_pattern_matches(patterns, shape_matches, max_results)

        if bottleneck_hints:
            bn_matches = self.query_by_bottleneck(bottleneck_hints, hardware_id, characteristics)
            patterns = _merge_pattern_matches(patterns, bn_matches, max_results)

        lessons = self.get_lessons(
            hardware_id,
            strategy_families=strategy_families,
            shape_regime_label=shape_point.label if shape_point else None,
            bottleneck_hints=bottleneck_hints,
        )

        suggested = _derive_suggestions(patterns)
        warnings = _derive_warnings(hardware_id, patterns, self._index)

        return QueryResult(
            patterns=patterns[:max_results],
            lessons=lessons,
            suggested_strategy_families=suggested,
            warnings=warnings,
        )

    def query_by_characteristics(
        self,
        characteristics: KernelCharacteristics,
        hardware_id: str,
        top_k: int = 5,
    ) -> list[PatternMatch]:
        candidates = self._candidate_patterns(hardware_id)
        matches: list[PatternMatch] = []
        for pattern in candidates:
            score, breakdown = _characteristics_score(characteristics, pattern)
            if score > 0:
                obs = _relevant_observations(characteristics, pattern)
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        score=score,
                        score_breakdown=breakdown,
                        matched_observations=obs,
                    )
                )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches[:top_k]

    def query_by_shape_regime(
        self,
        shape_point: ShapePoint,
        hardware_id: str,
        strategy_family: str | None = None,
    ) -> list[PatternMatch]:
        candidates = self._candidate_patterns(hardware_id, strategy_family)
        matches: list[PatternMatch] = []
        for pattern in candidates:
            score = _shape_regime_score(shape_point, pattern)
            if score > 0:
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        score=score,
                        score_breakdown={"shape": score},
                    )
                )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def query_by_bottleneck(
        self,
        bottleneck_hints: list[str],
        hardware_id: str,
        characteristics: KernelCharacteristics | None = None,
    ) -> list[PatternMatch]:
        candidates = self._candidate_patterns(hardware_id)
        hint_set = set(bottleneck_hints)
        matches: list[PatternMatch] = []
        for pattern in candidates:
            addressed = set(pattern.effect.bottlenecks_addressed)
            if not addressed:
                continue
            jaccard = len(hint_set & addressed) / len(hint_set | addressed)
            if jaccard > 0:
                matches.append(
                    PatternMatch(
                        pattern=pattern,
                        score=jaccard,
                        score_breakdown={"bottleneck": jaccard},
                    )
                )
        matches.sort(key=lambda m: m.score, reverse=True)
        return matches

    def get_lessons(
        self,
        hardware_id: str,
        strategy_families: list[str] | None = None,
        shape_regime_label: str | None = None,
        bottleneck_hints: list[str] | None = None,
    ) -> list[Lesson]:
        lessons: list[Lesson] = []
        for entry in self._index["lessons"].values():
            hw_ids: list[str] = entry.get("hardware_ids", [])
            hw_fams: list[str] = entry.get("hardware_families", [])
            if hw_ids and hardware_id not in hw_ids and not hw_fams:
                continue
            triggered = False
            if bottleneck_hints and set(entry.get("triggers_on_bottlenecks", [])) & set(
                bottleneck_hints
            ):
                triggered = True
            if strategy_families and set(entry.get("triggers_on_strategies", [])) & set(
                strategy_families
            ):
                triggered = True
            if shape_regime_label and shape_regime_label in entry.get(
                "triggers_on_shape_labels", []
            ):
                triggered = True
            if triggered:
                lesson = self._load_lesson(Path(entry["path"]))
                lessons.append(lesson)
        return lessons

    # ------------------------------------------------------------------
    # Direct lookups
    # ------------------------------------------------------------------

    def get_run(self, run_id: str) -> RunRecord:
        entry = self._index["runs"][run_id]
        with Path(entry["path"]).open() as f:
            return RunRecord.model_validate(json.load(f))

    def get_pattern(self, pattern_id: str) -> Pattern:
        entry = self._index["patterns"][pattern_id]
        return self._load_pattern(Path(entry["path"]))

    def get_lesson(self, lesson_id: str) -> Lesson:
        entry = self._index["lessons"][lesson_id]
        return self._load_lesson(Path(entry["path"]))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _candidate_patterns(
        self, hardware_id: str, strategy_family: str | None = None
    ) -> list[Pattern]:
        results: list[Pattern] = []
        for pid, entry in self._index["patterns"].items():
            if entry.get("hardware_id") != hardware_id:
                continue
            if strategy_family and entry.get("strategy_family") != strategy_family:
                continue
            results.append(self._load_pattern(Path(entry["path"])))
        return results

    def _load_pattern(self, path: Path) -> Pattern:
        text = path.read_text()
        front, body = _split_yaml_markdown(text)
        obs = [PatternObservation(**o) for o in front.pop("observations", [])]
        front.pop("schema_version", None)
        aw_raw = front.pop("applies_when", {})
        eff_raw = front.pop("effect", {})
        notes_raw = front.pop("implementation_notes", {})
        return Pattern(
            body_markdown=body,
            observations=obs,
            applies_when=PatternAppliesWhen.model_validate(aw_raw) if aw_raw else PatternAppliesWhen(),
            effect=PatternEffect.model_validate(eff_raw) if eff_raw else PatternEffect(),
            implementation_notes=PatternImplementationNotes.model_validate(notes_raw)
            if notes_raw
            else PatternImplementationNotes(),
            **front,
        )

    def _load_lesson(self, path: Path) -> Lesson:
        text = path.read_text()
        front, body = _split_yaml_markdown(text)
        front.pop("schema_version", None)
        tw_raw = front.pop("triggers_when", {})
        return Lesson(
            body_markdown=body,
            triggers_when=LessonTrigger.model_validate(tw_raw) if tw_raw else LessonTrigger(),
            **front,
        )

    @staticmethod
    def _bootstrap_pattern(
        strategy_family: str, hardware_id: str, run_record: RunRecord
    ) -> Pattern:
        return Pattern(
            id=f"pat_{strategy_family}_{hardware_id}",
            strategy_family=strategy_family,
            hardware_id=hardware_id,
            last_updated=date.today(),
            confidence=0.5,
            body_markdown=(
                f"## Strategy: {strategy_family.replace('-', ' ').title()} on {hardware_id}\n\n"
                "*Generated from first run. Edit to add context.*\n"
            ),
        )

    @staticmethod
    def _re_derive_aggregates(pattern: Pattern) -> Pattern:
        """Update aggregated fields from observations. Preserves body_markdown."""
        speedups = [
            o.speedup_in_regime
            for o in pattern.observations
            if o.speedup_in_regime is not None
        ]
        if speedups:
            pattern.effect = PatternEffect(
                bottlenecks_addressed=pattern.effect.bottlenecks_addressed,
                bottlenecks_not_addressed=pattern.effect.bottlenecks_not_addressed,
                typical_speedup_range=(min(speedups), max(speedups)),
                speedup_variance=_speedup_variance_label(speedups),
            )
        # Confidence: fraction of observations that produced a winning regime
        winner_count = sum(1 for o in pattern.observations if o.winning_regime is not None)
        if pattern.observations:
            pattern.confidence = round(
                0.5 + 0.5 * (winner_count / len(pattern.observations)), 2
            )
        return pattern

    def _write_pattern(self, pattern: Pattern, path: Path) -> None:
        front = self._pattern_to_front_matter(pattern)
        body = pattern.body_markdown
        _write_yaml_markdown(path, front, body)

    def _write_lesson(self, lesson: Lesson, path: Path) -> None:
        front = self._lesson_to_front_matter(lesson)
        body = lesson.body_markdown
        _write_yaml_markdown(path, front, body)

    @staticmethod
    def _pattern_to_front_matter(pattern: Pattern) -> dict[str, Any]:
        data = json.loads(pattern.model_dump_json(exclude={"body_markdown"}))
        return data

    @staticmethod
    def _lesson_to_front_matter(lesson: Lesson) -> dict[str, Any]:
        data = json.loads(lesson.model_dump_json(exclude={"body_markdown"}))
        return data


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _characteristics_score(
    chars: KernelCharacteristics, pattern: Pattern
) -> tuple[float, dict[str, float]]:
    """
    Composite score: 0.40 characteristics + 0.25 shape (skipped here) +
    0.25 bottleneck (skipped here) + 0.10 tags.
    Returns (score, breakdown).
    """
    aw = pattern.applies_when.kernel_characteristics

    # Tag overlap (Jaccard) — always computed, used as fallback when aw is empty
    all_tags: set[str] = set()
    for obs in pattern.observations:
        all_tags.update(obs.kernel_tags)
    query_tags = set(chars.characteristic_tags)
    tag_score = (
        len(query_tags & all_tags) / len(query_tags | all_tags)
        if query_tags or all_tags
        else 0.0
    )

    if not aw:
        # No structural requirements defined yet — score purely on tag overlap.
        # A bootstrapped pattern with matching tags is still a useful result.
        score = tag_score
        return score, {"tag_overlap": tag_score, "composite": score}

    # Weighted-vector threshold matching for memory access and compute structure
    mem_req = aw.get("memory_access_pattern_allows", {})
    comp_req = aw.get("compute_structure_requires", {})
    mem_score = _weighted_threshold_score(
        {k: getattr(chars.memory_access_pattern, k, 0.0) for k in mem_req}, mem_req
    )
    comp_score = _weighted_threshold_score(
        {k: getattr(chars.compute_structure, k, 0.0) for k in comp_req}, comp_req
    )

    # Arithmetic intensity class (exact = 1.0, adjacent = 0.5)
    ai_classes = aw.get("arithmetic_intensity_class", [])
    if ai_classes:
        ai_score = 1.0 if chars.arithmetic_intensity_class in ai_classes else 0.3
    else:
        ai_score = 1.0

    # Boolean flags
    bool_score = 1.0
    if aw.get("has_tensor_contractions") is not None:
        bool_score = 1.0 if chars.has_tensor_contractions == aw["has_tensor_contractions"] else 0.0

    chars_score = (mem_score + comp_score + ai_score + bool_score) / 4

    composite = 0.80 * chars_score + 0.20 * tag_score
    breakdown = {
        "memory_access": mem_score,
        "compute_structure": comp_score,
        "arithmetic_intensity": ai_score,
        "boolean_flags": bool_score,
        "tag_overlap": tag_score,
        "composite": composite,
    }
    return composite, breakdown


def _weighted_threshold_score(values: dict[str, float], requirements: dict[str, str]) -> float:
    if not requirements:
        return 1.0
    scores: list[float] = []
    for key, threshold_expr in requirements.items():
        threshold = _parse_threshold(threshold_expr)
        actual = values.get(key, 0.0)
        if actual >= threshold:
            scores.append(1.0)
        elif threshold > 0:
            scores.append(min(1.0, actual / threshold))
        else:
            scores.append(1.0)
    return sum(scores) / len(scores) if scores else 1.0


def _shape_regime_score(shape_point: ShapePoint, pattern: Pattern) -> float:
    """Score how well this shape_point falls in the pattern's winning regimes."""
    winning = pattern.applies_when.shape_conditions.get("winning_regimes", [])
    if not winning:
        return 0.0
    best = 0.0
    for condition in winning:
        score = _evaluate_shape_conditions(shape_point.dims, condition.conditions)
        best = max(best, score * condition.confidence)
    return best


def _evaluate_shape_conditions(dims: dict[str, int], conditions: dict[str, str]) -> float:
    if not conditions:
        return 0.5
    scores: list[float] = []
    for key, expr in conditions.items():
        if key not in dims:
            continue
        threshold = _parse_threshold(expr)
        actual = dims[key]
        op = expr.strip().split()[0] if expr.strip()[0].isalpha() else expr.strip()[:2]
        if ">=" in expr:
            scores.append(1.0 if actual >= threshold else min(1.0, actual / threshold))
        elif "<=" in expr:
            scores.append(1.0 if actual <= threshold else min(1.0, threshold / actual))
        elif ">" in expr:
            scores.append(1.0 if actual > threshold else min(1.0, actual / threshold))
        elif "<" in expr:
            scores.append(1.0 if actual < threshold else min(1.0, threshold / actual))
        else:
            scores.append(1.0 if actual == threshold else 0.0)
    return sum(scores) / len(scores) if scores else 0.5


def _relevant_observations(
    chars: KernelCharacteristics, pattern: Pattern
) -> list[PatternObservation]:
    query_tags = set(chars.characteristic_tags)
    scored = [
        (obs, len(query_tags & set(obs.kernel_tags)))
        for obs in pattern.observations
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [obs for obs, _ in scored[:3]]


def _merge_pattern_matches(
    a: list[PatternMatch], b: list[PatternMatch], top_k: int
) -> list[PatternMatch]:
    by_id: dict[str, PatternMatch] = {}
    for m in a:
        by_id[m.pattern.id] = m
    for m in b:
        if m.pattern.id in by_id:
            # merge: take max score, union breakdowns
            existing = by_id[m.pattern.id]
            merged_score = max(existing.score, m.score)
            merged_breakdown = {**existing.score_breakdown, **m.score_breakdown}
            by_id[m.pattern.id] = PatternMatch(
                pattern=existing.pattern,
                score=merged_score,
                score_breakdown=merged_breakdown,
                matched_observations=existing.matched_observations or m.matched_observations,
            )
        else:
            by_id[m.pattern.id] = m
    results = sorted(by_id.values(), key=lambda x: x.score, reverse=True)
    return results[:top_k]


def _derive_suggestions(matches: list[PatternMatch]) -> list[str]:
    seen: set[str] = set()
    suggestions: list[str] = []
    for m in matches:
        fam = m.pattern.strategy_family
        if fam not in seen:
            suggestions.append(fam)
            seen.add(fam)
    return suggestions[:5]


def _derive_warnings(
    hardware_id: str, matches: list[PatternMatch], index: dict[str, Any]
) -> list[str]:
    warnings: list[str] = []
    hw_patterns = [e for e in index["patterns"].values() if e.get("hardware_id") == hardware_id]
    if not hw_patterns:
        warnings.append(f"No prior experience found for hardware '{hardware_id}'.")
    if not matches:
        warnings.append("No matching patterns found; strategy suggestions will be LLM-only.")
    return warnings


# ---------------------------------------------------------------------------
# YAML / Markdown file utilities
# ---------------------------------------------------------------------------


def _split_yaml_markdown(text: str) -> tuple[dict[str, Any], str]:
    """Split a YAML front-matter + Markdown body file.
    Returns (front_matter_dict, markdown_body)."""
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    import io
    front: dict[str, Any] = yaml.load(io.StringIO(parts[1])) or {}
    body = parts[2].lstrip("\n")
    return front, body


def _write_yaml_markdown(path: Path, front: dict[str, Any], body: str) -> None:
    import io
    stream = io.StringIO()
    yaml.dump(front, stream)
    content = f"---\n{stream.getvalue()}---\n\n{body}"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def _parse_threshold(expr: str) -> float:
    """Extract numeric threshold from expressions like '>= 0.7' or '256'."""
    numbers = re.findall(r"[\d.]+", str(expr))
    return float(numbers[0]) if numbers else 0.0


def _speedup_variance_label(speedups: list[float]) -> str:
    if len(speedups) < 2:
        return "unknown"
    spread = max(speedups) - min(speedups)
    if spread < 0.3:
        return "low"
    elif spread < 1.0:
        return "medium"
    return "high"
