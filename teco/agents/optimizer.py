"""OptimizerAgent: the core optimization loop for TECO.

Flow:
  INIT   → profile baseline across shape sweep (Stage 1 + Stage 2)
         → query knowledge store for relevant patterns and lessons
         → extract kernel characteristics from code + profiling
  PLAN   → LLM generates N fundamentally different strategies
  LOOP   → each iteration:
               re-score / prune strategies by LLM confidence
               deepen leading strategy (produce code diff)
               apply diff → compile check → correctness validation
               Stage 1 profile across shape sweep
               update strategy shape results
  FINAL  → Stage 2 ncu on each regime winner
          → LLM synthesizes dispatch wrapper
          → log run record
"""

from __future__ import annotations

import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from teco.agents.base import BaseAgent
from teco.knowledge.schema import (
    KernelCharacteristics,
    PatternObservation,
    ProfilingReport,
    RunOutcome,
    RunRecord,
    ShapePoint,
    ShapeRange,
    Strategy,
    StrategyRecord,
    StrategyShapeResult,
    StrategyTree,
)
from teco.knowledge.store import KnowledgeStore
from teco.orchestration.context import OptimizationContext
from teco.tools.code_editor import apply_replacement, apply_unified_diff, generate_diff
from teco.tools.compiler import CompileResult, ValidationResult, compile_check, validate_correctness, extract_source_hash
from teco.tools.profiler import (
    HardwareCeilings,
    merge_stage1_into_report,
    profile_deep_ncu,
    profile_shape_sweep_stage1,
)

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert GPU kernel performance engineer specializing in {language} kernels.
    You have deep knowledge of GPU architecture, memory hierarchies, occupancy,
    tensor core utilization, and kernel optimization techniques.
    You reason carefully from profiling data to identify bottlenecks and apply targeted optimizations.
    Always respond with valid JSON when asked for structured output.
""")


class OptimizerAgent(BaseAgent):
    def __init__(
        self,
        store: KnowledgeStore,
        model_id: str = "claude-opus-4-6",
        ncu_output_dir: Path | None = None,
    ) -> None:
        super().__init__(model_id=model_id)
        self.store = store
        self.ncu_output_dir = ncu_output_dir or Path("knowledge/runs/ncu")

    def run(self, context: OptimizationContext) -> OptimizationContext:
        context.log(f"[OptimizerAgent] Starting optimization: {context.task_name}")

        # ── INIT ─────────────────────────────────────────────────────────────
        context = self._init_phase(context)

        # ── STRATEGY PLANNING ────────────────────────────────────────────────
        context = self._planning_phase(context)

        # ── ITERATION LOOP ───────────────────────────────────────────────────
        context = self._iteration_loop(context)

        # ── FINALISE ─────────────────────────────────────────────────────────
        context = self._finalise(context)

        return context

    # ------------------------------------------------------------------
    # INIT phase
    # ------------------------------------------------------------------

    def _init_phase(self, context: OptimizationContext) -> OptimizationContext:
        context.log("[OptimizerAgent] Phase: INIT — profiling baseline")

        # Show initial kernel source (level 2)
        src_lines = context.kernel_source.splitlines()
        if context.verbosity >= 2:
            max_lines = 80
            shown = src_lines[:max_lines]
            context.log(f"\n[Initial kernel source — {context.kernel_path.name}]")
            context.log("\n".join(shown), level=2)
            if len(src_lines) > max_lines:
                context.log(f"  ... ({len(src_lines)} lines total)", level=2)

        # Stage 1: latency across shape sweep
        baseline_s1: dict[str, dict[str, float]] = {}
        for shape in context.shape_sweep:
            # The agent will generate a make_fn via tool call in practice;
            # here we define the protocol — runner scripts handle actual execution
            metrics = self._profile_stage1_subprocess(
                context.kernel_source, shape, context
            )
            baseline_s1[shape.label] = metrics

        context.baseline_stage1 = baseline_s1

        # Stage 2: deep ncu at medium shape (most representative)
        medium_shape = next(
            (s for s in context.shape_sweep if s.label == "medium"),
            context.shape_sweep[0] if context.shape_sweep else None,
        )
        if medium_shape is not None:
            report = self._profile_stage2(
                context.kernel_source, medium_shape, context, label="baseline_medium"
            )
            if report is not None:
                # Merge Stage 1 metrics into Stage 2 report
                s1 = baseline_s1.get(medium_shape.label, {})
                flops, nbytes = self._estimate_flops_bytes(context, medium_shape)
                report = merge_stage1_into_report(report, s1, flops, nbytes)
                context.baseline_stage2[medium_shape.label] = report
                context.log(_fmt_report(report), level=3)

        # Print baseline performance table (level 1)
        context.log(_fmt_baseline_table(context), level=1)

        # Extract kernel characteristics from code + profiling
        context.kernel_characteristics = self._extract_characteristics(context)

        # Query knowledge store
        hw_id = _make_hardware_id(context.ceilings.device_name)
        medium_sp = medium_shape or ShapePoint(label="medium", dims={})
        bn_hints = [context.primary_bottleneck()] if context.primary_bottleneck() != "unknown" else []
        context.knowledge_query = self.store.query(
            characteristics=context.kernel_characteristics,
            hardware_id=hw_id,
            shape_point=medium_sp,
            bottleneck_hints=bn_hints,
        )

        if context.knowledge_query.warnings:
            for w in context.knowledge_query.warnings:
                context.log(f"  [knowledge] WARNING: {w}", level=1)

        return context

    # ------------------------------------------------------------------
    # PLANNING phase
    # ------------------------------------------------------------------

    def _planning_phase(self, context: OptimizationContext) -> OptimizationContext:
        context.log("[OptimizerAgent] Phase: STRATEGY PLANNING")

        system = _SYSTEM_PROMPT.format(language=context.language)
        user = self._build_planning_prompt(context)
        response = self._llm_call(system, user, max_tokens=6000)

        strategies = self._parse_strategies(response)
        tree = StrategyTree(
            strategies=strategies,
            shape_sweep=context.shape_sweep,
            active_ids=[s.id for s in strategies],
            iteration=0,
        )
        context.strategy_tree = tree

        context.log(_fmt_strategy_table(strategies), level=2)

        return context

    # ------------------------------------------------------------------
    # ITERATION LOOP
    # ------------------------------------------------------------------

    def _iteration_loop(self, context: OptimizationContext) -> OptimizationContext:
        consecutive_plateau = 0

        for iteration in range(context.max_iterations):
            context.iteration = iteration
            context.strategy_tree.iteration = iteration
            context.log(f"\n[OptimizerAgent] Iteration {iteration + 1}/{context.max_iterations}")

            # 1. Re-score and prune strategies
            context = self._rescore_strategies(context)

            active = [
                s for s in context.strategy_tree.strategies if s.status == "active"
            ]
            if not active:
                context.log("  All strategies resolved (winners + pruned). Done.")
                break

            # 2. Deepen the leading strategy
            leader = max(active, key=lambda s: s.llm_confidence)
            context.log(f"  Leading strategy: [{leader.id}] {leader.name} (confidence {leader.llm_confidence:.2f})")

            new_source, diff = self._deepen_strategy(leader, context)
            if not new_source:
                context.log(f"  Failed to generate new code for [{leader.id}]. Skipping iteration.")
                continue

            # Show proposed diff before compile (level 2)
            if diff and context.verbosity >= 2:
                diff_lines = diff.splitlines()
                cap = 60
                shown = "\n".join(diff_lines[:cap])
                context.log(f"\n  [{leader.id}] Proposed changes (iteration {iteration + 1}):", level=2)
                context.log(shown, level=2)
                if len(diff_lines) > cap:
                    context.log(f"  ... ({len(diff_lines) - cap} more lines)", level=2)

            # 3. Compile check
            compile_result = compile_check(new_source)
            if not compile_result.success:
                msg = compile_result.error_message[:400]
                context.log(f"  Compile check failed: {msg}")
                leader.llm_confidence = max(0.0, leader.llm_confidence - 0.15)
                leader.last_failure = f"Compile error:\n{msg}"
                continue

            # 4. Correctness validation
            val_result = validate_correctness(context.kernel_source, new_source)
            if not val_result.success:
                msg = val_result.error_message[:600]
                context.log(f"  Correctness validation failed: {msg}")
                leader.llm_confidence = max(0.0, leader.llm_confidence - 0.20)
                leader.last_failure = f"Correctness validation failed:\n{msg}"
                continue

            leader.last_failure = ""  # clear on success

            # 5. Stage 1 profile across shape sweep
            shape_results = self._profile_strategy_stage1(new_source, context)
            leader.shape_results = shape_results
            leader.current_code = new_source
            leader.iterations_applied += 1

            # Report per-shape progress
            best_tflops = max((r.achieved_tflops for r in shape_results), default=0.0)
            context.log(f"  Best TFLOPS this iteration: {best_tflops:.1f}")
            context.log(_fmt_shape_results(shape_results), level=2)

            # 6. Check efficiency ceiling
            eff = context.efficiency_pct(best_tflops)
            if eff >= context.target_efficiency_pct:
                context.log(f"  Reached {eff:.1f}% of hardware ceiling. Stopping.")
                break

            # 7. Plateau detection
            prev_best = context.best_tflops_across_shapes()
            if best_tflops <= prev_best * (1 + context.plateau_threshold):
                consecutive_plateau += 1
            else:
                consecutive_plateau = 0
            if consecutive_plateau >= 2:
                context.log("  Plateau detected (2 consecutive iterations < 2% improvement). Stopping.")
                break

        return context

    # ------------------------------------------------------------------
    # FINALISE
    # ------------------------------------------------------------------

    def _finalise(self, context: OptimizationContext) -> OptimizationContext:
        context.log("\n[OptimizerAgent] Phase: FINALISE")

        tree = context.strategy_tree
        # Mark regime winners from strategies with shape results
        for strategy in tree.strategies:
            if strategy.shape_results and strategy.status == "active":
                # Determine if this strategy wins any regime
                regime = self._determine_winning_regime(strategy, context)
                if regime:
                    strategy.status = "regime_winner"
                    strategy.winning_shape_range = regime
                    tree.regime_winners[strategy.id] = regime.description

        # Stage 2 ncu on each regime winner at its best shape
        regime_winners = [s for s in tree.strategies if s.status == "regime_winner"]
        for strategy in regime_winners:
            best_shape = self._best_shape_for_strategy(strategy, context)
            report = self._profile_stage2(
                strategy.current_code or context.kernel_source,
                best_shape,
                context,
                label=f"{context.run_id}_{strategy.id}_{best_shape.label}",
            )
            if report and strategy.shape_results:
                context.log(_fmt_report(report), level=3)
                # Attach to matching result
                for result in strategy.shape_results:
                    if result.shape_point.label == best_shape.label:
                        result.profiling_report = report

        # Generate dispatch wrapper if multiple regime winners
        if len(regime_winners) > 1:
            dispatch_logic = self._generate_dispatch_wrapper(regime_winners, context)
        elif len(regime_winners) == 1:
            dispatch_logic = f"# Single regime winner: {regime_winners[0].name}\n{regime_winners[0].current_code}"
        else:
            dispatch_logic = None

        # Validate dispatch wrapper
        if dispatch_logic and len(regime_winners) > 1:
            val = validate_correctness(context.kernel_source, dispatch_logic)
            if not val.success:
                context.log(f"  WARNING: dispatch wrapper validation failed: {val.error_message[:200]}")

        # Compute outcome
        overall_speedup = self._compute_overall_speedup(context)
        regime_speedups = self._compute_regime_speedups(context)
        outcome = RunOutcome(
            overall_speedup=overall_speedup,
            regime_speedups=regime_speedups,
            total_iterations=context.iteration + 1,
            strategies_tried=len(tree.strategies),
            strategies_pruned=sum(1 for s in tree.strategies if s.status == "pruned"),
            strategies_deployed=len(regime_winners),
        )

        # Build and write run record
        hw_id = _make_hardware_id(context.ceilings.device_name)
        from teco.knowledge.schema import BaselineProfile, HardwareInfo
        hw_info = HardwareInfo(
            device_name=context.ceilings.device_name,
            device_id=hw_id,
            peak_tflops_fp16=context.ceilings.peak_tflops_fp16,
            peak_tflops_fp32=context.ceilings.peak_tflops_fp32,
            peak_bandwidth_gbs=context.ceilings.peak_bandwidth_gbs,
            sm_count=0,
            compute_capability="",
        )
        run_record = RunRecord(
            id=context.run_id,
            created_at=datetime.now(timezone.utc),
            task=context.task_name,
            hardware=hw_info,
            language=context.language,
            kernel_characteristics=context.kernel_characteristics or KernelCharacteristics(),
            shape_sweep=context.shape_sweep,
            baseline=BaselineProfile(
                source_hash=extract_source_hash(context.kernel_source),
                profiling_by_shape={
                    label: metrics
                    for label, metrics in context.baseline_stage1.items()
                },
            ),
            strategies=self._strategy_records(tree),
            dispatch_logic=dispatch_logic,
            outcome=outcome,
        )

        self.store.write_run(run_record)

        # Update pattern files for all strategies
        for strategy in tree.strategies:
            obs = PatternObservation(
                run_id=run_record.id,
                task=run_record.task,
                kernel_tags=context.kernel_characteristics.characteristic_tags
                if context.kernel_characteristics
                else [],
                winning_regime=strategy.winning_shape_range.description
                if strategy.winning_shape_range
                else None,
                speedup_in_regime=self._strategy_speedup(strategy, context),
                note=strategy.prune_reason or f"Status: {strategy.status}",
            )
            self.store.upsert_pattern(
                strategy_family=strategy.family,
                hardware_id=hw_id,
                new_observation=obs,
                run_record=run_record,
            )

        context.log(f"\n[OptimizerAgent] Done. Overall speedup: {overall_speedup:.2f}x")
        context.log(f"  Run record: knowledge/runs/{run_record.id}.json")
        return context

    # ------------------------------------------------------------------
    # LLM prompt builders
    # ------------------------------------------------------------------

    def _build_planning_prompt(self, context: OptimizationContext) -> str:
        chars = context.kernel_characteristics
        baseline_summary = json.dumps(context.baseline_stage1, indent=2)
        stage2_summary = ""
        if context.baseline_stage2:
            rep = next(iter(context.baseline_stage2.values()))
            stage2_summary = (
                f"\n## Deep profiling (ncu) at medium shape\n"
                f"- Bottleneck: {rep.bottleneck}\n"
                f"- Tensor core utilization: {rep.tensor_core_utilization:.1%}\n"
                f"- Achieved occupancy: {rep.achieved_occupancy:.1%} "
                f"(theoretical: {rep.theoretical_occupancy:.1%})\n"
                f"- L1 hit rate: {rep.l1_hit_rate:.1%}, L2 hit rate: {rep.l2_hit_rate:.1%}\n"
                f"- Warp efficiency: {rep.warp_efficiency:.1%}\n"
                f"- Branch divergence: {rep.branch_divergence_pct:.1f}%\n"
                f"- Shared memory bank conflicts: {rep.shared_mem_bank_conflicts}\n"
            )

        knowledge_summary = ""
        if context.knowledge_query:
            if context.knowledge_query.lessons:
                knowledge_summary += "\n## Relevant lessons from experience\n"
                for lesson in context.knowledge_query.lessons[:3]:
                    knowledge_summary += f"- {lesson.title} (confidence {lesson.confidence:.0%})\n"
            if context.knowledge_query.patterns:
                knowledge_summary += "\n## Relevant past strategy patterns\n"
                for match in context.knowledge_query.patterns[:4]:
                    p = match.pattern
                    knowledge_summary += (
                        f"- **{p.strategy_family}** (score {match.score:.2f}): "
                        f"typical speedup {p.effect.typical_speedup_range[0]:.1f}–"
                        f"{p.effect.typical_speedup_range[1]:.1f}x\n"
                    )

        return textwrap.dedent(f"""\
            ## Task
            Kernel: {context.task_name} ({context.language})
            Hardware: {context.ceilings.device_name}
            Peak FP16 TFLOPS: {context.ceilings.peak_tflops_fp16:.0f} | Peak BW: {context.ceilings.peak_bandwidth_gbs:.0f} GB/s

            ## Kernel source
            ```python
            {context.kernel_source[:6000]}
            ```

            ## Kernel characteristics (extracted)
            - Memory access: coalesced={chars.memory_access_pattern.coalesced:.1f}, random={chars.memory_access_pattern.random:.1f}, gather={chars.memory_access_pattern.gather_scatter:.1f}
            - Compute: tensor_core={chars.compute_structure.tensor_core_parallelizable:.1f}, reduction={chars.compute_structure.reduction:.1f}
            - Arithmetic intensity class: {chars.arithmetic_intensity_class}
            - Tags: {', '.join(chars.characteristic_tags)}
            {stage2_summary}
            ## Baseline performance (Stage 1, across shape sweep)
            ```json
            {baseline_summary}
            ```
            {knowledge_summary}

            ## Your task
            Generate {3}–{5} fundamentally DIFFERENT high-level optimization strategies for this kernel.
            Each strategy must represent a different approach that is largely INCOMPATIBLE with the others
            (e.g., you cannot apply both "tensor core rewrite" and "scalar vectorization" in the same kernel
            without one overriding the other). For each strategy, also predict which shape regime
            (small/medium/large) you expect it to win.

            Respond with valid JSON in this exact format:
            {{
              "strategies": [
                {{
                  "id": "s1",
                  "name": "short-slug-name",
                  "family": "strategy-family-slug",
                  "rationale": "Why this approach addresses the bottleneck",
                  "addresses_bottleneck": "specific bottleneck from ncu data",
                  "implementation_sketch": "Key code changes needed (2-4 sentences)",
                  "predicted_winning_regime": "e.g. 'M >= 512, N power-of-two'",
                  "incompatible_with": ["s2"],
                  "llm_confidence": 0.8
                }}
              ]
            }}
        """)

    def _build_deepen_prompt(self, strategy: Strategy, context: OptimizationContext) -> str:
        shape_perf = ""
        if strategy.shape_results:
            rows = "\n".join(
                f"  {r.shape_point.label}: {r.achieved_tflops:.1f} TFLOPS "
                f"({r.vs_baseline_pct:+.1f}% vs baseline)"
                for r in strategy.shape_results
            )
            shape_perf = f"\n## Current performance of this strategy\n{rows}\n"

        failure_section = ""
        if strategy.last_failure:
            failure_section = textwrap.dedent(f"""\

                ## IMPORTANT: previous attempt was rejected
                Your last generated code was rejected with this error — you MUST fix it:
                ```
                {strategy.last_failure[:800]}
                ```
                Diagnose the root cause and produce a corrected version.
            """)

        return textwrap.dedent(f"""\
            ## Strategy to implement / refine
            Name: {strategy.name}
            Rationale: {strategy.rationale}
            Addresses: {strategy.addresses_bottleneck}
            Predicted winning regime: {strategy.predicted_winning_regime}
            Implementation sketch: {strategy.implementation_sketch}
            Iteration: {strategy.iterations_applied + 1}
            {shape_perf}{failure_section}
            ## Current kernel source
            ```python
            {strategy.current_code or context.kernel_source}
            ```

            ## Task
            Produce the next concrete improvement for this strategy. Return the complete
            optimized kernel source (not a diff) in a ```python ... ``` code block.
            Keep the test function (after the {'#' * 10}... separator) unchanged.
            Explain your change in 2-3 sentences BEFORE the code block.
        """)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_characteristics(self, context: OptimizationContext) -> KernelCharacteristics:
        """Ask LLM to extract KernelCharacteristics from source + profiling."""
        system = _SYSTEM_PROMPT.format(language=context.language)
        stage2_info = ""
        if context.baseline_stage2:
            rep = next(iter(context.baseline_stage2.values()))
            stage2_info = (
                f"Bottleneck: {rep.bottleneck}, "
                f"tensor_core_util={rep.tensor_core_utilization:.1%}, "
                f"l1_hit={rep.l1_hit_rate:.1%}, l2_hit={rep.l2_hit_rate:.1%}, "
                f"warp_eff={rep.warp_efficiency:.1%}"
            )

        user = textwrap.dedent(f"""\
            Analyze this {context.language} kernel and extract its structural characteristics.

            ## Kernel source
            ```python
            {context.kernel_source[:5000]}
            ```

            ## Profiling summary (medium shape)
            {stage2_info or 'Not available.'}

            Respond with valid JSON matching this schema:
            {{
              "memory_access_pattern": {{"coalesced": 0.0, "strided": 0.0, "random": 0.0, "gather_scatter": 0.0}},
              "compute_structure": {{"tensor_core_parallelizable": 0.0, "vector_parallelizable": 0.0, "reduction": 0.0, "scan": 0.0, "irregular": 0.0}},
              "arithmetic_intensity_class": "low|medium|high",
              "data_types": ["fp16"],
              "dominant_data_type": "fp16",
              "has_shared_memory_reuse": false,
              "has_tensor_contractions": false,
              "has_irregular_control_flow": false,
              "parallelism_axes": [],
              "reduction_axes": [],
              "characteristic_tags": [],
              "extraction_confidence": 0.7
            }}
        """)
        response = self._llm_call(system, user, max_tokens=2048)
        try:
            data = json.loads(_extract_json(response))
            return KernelCharacteristics.model_validate(data)
        except Exception:
            return KernelCharacteristics()

    def _rescore_strategies(self, context: OptimizationContext) -> OptimizationContext:
        """Ask LLM to update confidence scores and prune dominated strategies."""
        tree = context.strategy_tree
        active = [s for s in tree.strategies if s.status == "active"]
        if not active:
            return context

        perf_summary = self._format_strategy_performance(tree)
        system = _SYSTEM_PROMPT.format(language=context.language)
        user = textwrap.dedent(f"""\
            ## Current optimization state (iteration {context.iteration + 1})
            Hardware: {context.ceilings.device_name}
            Shape sweep labels: {[s.label for s in context.shape_sweep]}

            ## Strategy performance so far
            {perf_summary}

            ## Task
            Re-score each ACTIVE strategy. For each strategy, determine:
            1. Updated confidence (0–1) based on measured performance across shapes.
            2. Whether to PRUNE it (if it is dominated in ALL shape regimes by other strategies).
            3. Whether to mark as REGIME_WINNER (if it clearly leads for a specific shape range).

            Respond with valid JSON:
            {{
              "updates": [
                {{
                  "id": "s1",
                  "llm_confidence": 0.85,
                  "status": "active|pruned|regime_winner",
                  "winning_shape_range_description": "seq_q >= 512",
                  "winning_shape_range_conditions": {{"seq_q": ">= 512"}},
                  "prune_reason": null
                }}
              ]
            }}
        """)
        response = self._llm_call(system, user, max_tokens=2048)

        try:
            data = json.loads(_extract_json(response))
            for update in data.get("updates", []):
                strategy = next(
                    (s for s in tree.strategies if s.id == update["id"]), None
                )
                if strategy is None or strategy.status not in ("active",):
                    continue
                strategy.llm_confidence = float(update.get("llm_confidence", strategy.llm_confidence))
                new_status = update.get("status", "active")
                if new_status == "pruned":
                    strategy.status = "pruned"
                    strategy.prune_reason = update.get("prune_reason", "LLM decided to prune")
                    tree.active_ids = [i for i in tree.active_ids if i != strategy.id]
                elif new_status == "regime_winner":
                    strategy.status = "regime_winner"
                    desc = update.get("winning_shape_range_description", "")
                    conds = update.get("winning_shape_range_conditions", {})
                    if desc:
                        strategy.winning_shape_range = ShapeRange(
                            description=desc, conditions=conds
                        )
                    tree.regime_winners[strategy.id] = desc
                    tree.active_ids = [i for i in tree.active_ids if i != strategy.id]
        except Exception as e:
            context.log(f"  [rescore] Warning: could not parse response: {e}", level=2)

        return context

    def _deepen_strategy(
        self, strategy: Strategy, context: OptimizationContext
    ) -> tuple[str, str]:
        """Ask LLM to produce next implementation of the strategy. Returns (new_source, diff)."""
        system = _SYSTEM_PROMPT.format(language=context.language)
        user = self._build_deepen_prompt(strategy, context)
        response = self._llm_call(system, user, max_tokens=8192)

        new_source = _extract_code_block(response)
        if not new_source:
            return "", ""

        original = strategy.current_code or context.kernel_source
        diff = generate_diff(original, new_source)
        return new_source, diff

    def _profile_strategy_stage1(
        self, source: str, context: OptimizationContext
    ) -> list[StrategyShapeResult]:
        """Profile a candidate kernel across the shape sweep using Stage 1."""
        results: list[StrategyShapeResult] = []
        for shape in context.shape_sweep:
            metrics = self._profile_stage1_subprocess(source, shape, context)
            tflops = metrics.get("tflops", 0.0)
            baseline_tflops = context.baseline_stage1.get(shape.label, {}).get("tflops", 0.0)
            vs_baseline = (
                ((tflops - baseline_tflops) / baseline_tflops * 100)
                if baseline_tflops > 0
                else 0.0
            )
            results.append(
                StrategyShapeResult(
                    shape_point=shape,
                    achieved_tflops=tflops,
                    vs_baseline_pct=vs_baseline,
                )
            )
        return results

    def _profile_stage1_subprocess(
        self, source: str, shape: ShapePoint, context: OptimizationContext
    ) -> dict[str, float]:
        """Run Stage 1 profiling by executing a generated runner script in a subprocess."""
        import subprocess
        import sys
        import tempfile

        runner = self._generate_runner_script(source, shape, context, stage=1)
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write(runner)
            runner_path = Path(f.name)

        try:
            result = subprocess.run(
                [sys.executable, str(runner_path)],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                return json.loads(result.stdout.strip())
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
            pass
        finally:
            runner_path.unlink(missing_ok=True)
        return {"tflops": 0.0, "latency_ms": 0.0}

    def _profile_stage2(
        self,
        source: str,
        shape: ShapePoint,
        context: OptimizationContext,
        label: str,
    ):
        """Run Stage 2 ncu profiling for a single (kernel, shape) pair."""
        runner = self._generate_runner_script(source, shape, context, stage=2)
        return profile_deep_ncu(
            runner_script=runner,
            output_dir=self.ncu_output_dir,
            label=label,
            ceilings=context.ceilings,
        )

    def _generate_runner_script(
        self, source: str, shape: ShapePoint, context: OptimizationContext, stage: int
    ) -> str:
        """Generate a standalone Python script that executes the kernel at a given shape.

        For Stage 1: prints JSON with {tflops, latency_ms, bandwidth_gbs}.
        For Stage 2: just runs the kernel once (ncu wraps the process).

        The LLM is asked to generate the benchmark harness specific to the kernel.
        """
        # In practice the agent generates the runner; this is the fallback for
        # kernels that follow the TritonBench test function convention.
        dims_str = ", ".join(f"{k}={v}" for k, v in shape.dims.items())
        if stage == 1:
            return textwrap.dedent(f"""\
                import json, sys, torch
                sys.path.insert(0, '.')

                # --- Kernel source ---
                {source}

                # --- Benchmark harness ---
                try:
                    from triton.testing import do_bench
                    # Attempt to call the first test_ function and time it
                    import inspect
                    test_fns = [(k, v) for k, v in list(globals().items())
                                if k.startswith('test_') and callable(v)]
                    if test_fns:
                        fn_name, fn = test_fns[0]
                        latency_ms = do_bench(fn, warmup=10, rep=50)
                        print(json.dumps({{"tflops": 0.0, "latency_ms": latency_ms}}))
                    else:
                        print(json.dumps({{"tflops": 0.0, "latency_ms": 0.0}}))
                except Exception as e:
                    print(json.dumps({{"error": str(e), "tflops": 0.0, "latency_ms": 0.0}}))
            """)
        else:  # stage 2: just run
            return textwrap.dedent(f"""\
                import sys, torch
                sys.path.insert(0, '.')

                # --- Kernel source ---
                {source}

                # --- Single execution for ncu ---
                try:
                    import inspect
                    test_fns = [(k, v) for k, v in list(globals().items())
                                if k.startswith('test_') and callable(v)]
                    if test_fns:
                        test_fns[0][1]()
                except Exception as e:
                    print(f"ERROR: {{e}}", file=sys.stderr)
            """)

    def _generate_dispatch_wrapper(
        self, winners: list[Strategy], context: OptimizationContext
    ) -> str:
        """Ask LLM to synthesize a dispatch wrapper from multiple regime winners."""
        strategy_info = "\n".join(
            f"  - Strategy '{s.name}' wins for: {s.winning_shape_range.description if s.winning_shape_range else 'unknown'}"
            for s in winners
        )
        system = _SYSTEM_PROMPT.format(language=context.language)
        user = textwrap.dedent(f"""\
            ## Regime-winning strategies
            {strategy_info}

            ## Strategy implementations
            {chr(10).join(f"### {s.name}{chr(10)}```python{chr(10)}{s.current_code[:3000]}{chr(10)}```" for s in winners)}

            ## Task
            Generate a dispatch wrapper that selects the appropriate implementation at runtime
            based on input tensor shapes. The wrapper should:
            1. Inspect the relevant shape parameters
            2. Call the appropriate strategy implementation
            3. Be a drop-in replacement for the original kernel entry point

            Return the complete source (all implementations + dispatch wrapper) in ```python ... ```.
        """)
        response = self._llm_call(system, user, max_tokens=12000)
        return _extract_code_block(response) or winners[-1].current_code

    def _estimate_flops_bytes(
        self, context: OptimizationContext, shape: ShapePoint
    ) -> tuple[int, int]:
        """Return (flops, bytes) estimate for a shape point. Placeholder — LLM fills this."""
        # Rough heuristic: for attention-like kernels; will be overridden by kernel-specific logic
        dims = shape.dims
        m = dims.get("M", dims.get("seq_q", 512))
        n = dims.get("N", dims.get("seq_k", 512))
        k = dims.get("K", dims.get("head_dim", 64))
        flops = 2 * m * n * k  # approximate matmul FLOPs
        nbytes = 2 * (m * k + k * n + m * n) * 2  # fp16 = 2 bytes
        return flops, nbytes

    def _determine_winning_regime(
        self, strategy: Strategy, context: OptimizationContext
    ) -> ShapeRange | None:
        """Return a ShapeRange if this strategy's results show it leading for some shapes."""
        if not strategy.shape_results:
            return None
        winning_labels = [
            r.shape_point.label for r in strategy.shape_results if r.vs_baseline_pct > 5.0
        ]
        if not winning_labels:
            return None
        description = f"shape in {{{', '.join(winning_labels)}}}"
        return ShapeRange(description=description, conditions={"regime": f"in {winning_labels}"})

    def _best_shape_for_strategy(
        self, strategy: Strategy, context: OptimizationContext
    ) -> ShapePoint:
        if strategy.shape_results:
            best = max(strategy.shape_results, key=lambda r: r.achieved_tflops)
            return best.shape_point
        return context.shape_sweep[len(context.shape_sweep) // 2] if context.shape_sweep else ShapePoint(label="medium", dims={})

    def _compute_overall_speedup(self, context: OptimizationContext) -> float:
        baseline = context.baseline_stage1.get("medium", {}).get("tflops", 0.0)
        if baseline <= 0:
            return 1.0
        best = context.best_tflops_across_shapes()
        return best / baseline if baseline > 0 else 1.0

    def _compute_regime_speedups(self, context: OptimizationContext) -> dict[str, float]:
        speedups: dict[str, float] = {}
        for shape in context.shape_sweep:
            baseline = context.baseline_stage1.get(shape.label, {}).get("tflops", 0.0)
            if baseline <= 0:
                continue
            best = max(
                (
                    next(
                        (r.achieved_tflops for r in s.shape_results if r.shape_point.label == shape.label),
                        0.0,
                    )
                    for s in context.strategy_tree.strategies
                    if s.status == "regime_winner"
                ),
                default=baseline,
            )
            speedups[shape.label] = best / baseline
        return speedups

    def _strategy_speedup(self, strategy: Strategy, context: OptimizationContext) -> float | None:
        if not strategy.shape_results:
            return None
        best_result = max(strategy.shape_results, key=lambda r: r.achieved_tflops)
        baseline = context.baseline_stage1.get(best_result.shape_point.label, {}).get("tflops", 0.0)
        if baseline <= 0:
            return None
        return best_result.achieved_tflops / baseline

    def _strategy_records(self, tree: StrategyTree) -> list[StrategyRecord]:
        records: list[StrategyRecord] = []
        for s in tree.strategies:
            shape_results = {
                r.shape_point.label: {
                    "achieved_tflops": r.achieved_tflops,
                    "vs_baseline_pct": r.vs_baseline_pct,
                }
                for r in s.shape_results
            }
            final_ncu: dict[str, float] | None = None
            if s.shape_results:
                for r in s.shape_results:
                    if r.profiling_report is not None:
                        rp = r.profiling_report
                        final_ncu = {
                            "tensor_core_utilization": rp.tensor_core_utilization,
                            "achieved_occupancy": rp.achieved_occupancy,
                            "warp_efficiency": rp.warp_efficiency,
                            "shared_mem_bank_conflicts": float(rp.shared_mem_bank_conflicts),
                        }
                        break
            records.append(
                StrategyRecord(
                    id=s.id,
                    name=s.name,
                    family=s.family,
                    status=s.status,
                    winning_shape_range=s.winning_shape_range,
                    prune_reason=s.prune_reason,
                    predicted_winning_regime=s.predicted_winning_regime,
                    llm_confidence_final=s.llm_confidence,
                    addresses_bottleneck=s.addresses_bottleneck,
                    incompatible_with=s.incompatible_with,
                    iterations_applied=s.iterations_applied,
                    shape_results=shape_results,
                    final_ncu=final_ncu,
                )
            )
        return records

    def _format_strategy_performance(self, tree: StrategyTree) -> str:
        lines: list[str] = []
        for s in tree.strategies:
            if s.status == "pruned":
                lines.append(f"[{s.id}] {s.name} — PRUNED: {s.prune_reason}")
                continue
            if s.status == "regime_winner":
                lines.append(
                    f"[{s.id}] {s.name} — REGIME WINNER: {s.winning_shape_range.description if s.winning_shape_range else '?'}"
                )
                continue
            perf = ""
            if s.shape_results:
                perf = " | ".join(
                    f"{r.shape_point.label}: {r.achieved_tflops:.1f} TFLOPS ({r.vs_baseline_pct:+.1f}%)"
                    for r in s.shape_results
                )
            lines.append(
                f"[{s.id}] {s.name} (confidence {s.llm_confidence:.2f}) — {perf or 'no profiling yet'}"
            )
        return "\n".join(lines)

    def _parse_strategies(self, response: str) -> list[Strategy]:
        try:
            data = json.loads(_extract_json(response))
            strategies: list[Strategy] = []
            for item in data.get("strategies", []):
                strategies.append(
                    Strategy(
                        id=item.get("id", f"s{len(strategies)+1}"),
                        name=item.get("name", "unnamed"),
                        family=item.get("family", item.get("name", "unknown")),
                        rationale=item.get("rationale", ""),
                        addresses_bottleneck=item.get("addresses_bottleneck", ""),
                        implementation_sketch=item.get("implementation_sketch", ""),
                        predicted_winning_regime=item.get("predicted_winning_regime", "unknown"),
                        incompatible_with=item.get("incompatible_with", []),
                        llm_confidence=float(item.get("llm_confidence", 0.5)),
                    )
                )
            return strategies
        except Exception:
            # Fallback: return a single generic strategy
            return [
                Strategy(
                    id="s1",
                    name="generic-optimization",
                    family="generic",
                    rationale="LLM strategy parsing failed; using fallback",
                    addresses_bottleneck="unknown",
                    implementation_sketch="Apply general GPU optimizations",
                    predicted_winning_regime="all shapes",
                    llm_confidence=0.3,
                )
            ]


# ---------------------------------------------------------------------------
# Verbosity formatting helpers
# ---------------------------------------------------------------------------


def _fmt_baseline_table(context: "OptimizationContext") -> str:  # type: ignore[name-defined]
    """Format baseline Stage 1 performance as a compact table (level 1)."""
    lines = ["\n[Baseline performance]",
             f"  {'Shape':<10}  {'Latency (ms)':>12}  {'TFLOPS':>8}"]
    lines.append("  " + "─" * 34)
    for shape in context.shape_sweep:
        m = context.baseline_stage1.get(shape.label, {})
        lat = m.get("latency_ms", 0.0)
        tfl = m.get("tflops", 0.0)
        lines.append(f"  {shape.label:<10}  {lat:>12.2f}  {tfl:>8.1f}")
    # Append Stage 2 bottleneck summary if available
    rep = context.baseline_stage2.get("medium") or (
        next(iter(context.baseline_stage2.values()), None) if context.baseline_stage2 else None
    )
    if rep is not None:
        lines.append(
            f"  Bottleneck (medium): {rep.bottleneck}"
            f"  |  tensor-core util: {rep.tensor_core_utilization:.1%}"
            f"  |  occupancy: {rep.achieved_occupancy:.1%}"
        )
        lines.append(
            f"  DRAM: {rep.dram_bandwidth_gbs:.0f} GB/s (peak {rep.peak_bandwidth_gbs:.0f})"
            f"  |  L1: {rep.l1_hit_rate:.0%}  |  L2: {rep.l2_hit_rate:.0%}"
            f"  |  warp eff: {rep.warp_efficiency:.0%}"
        )
    return "\n".join(lines)


def _fmt_strategy_table(strategies: "list[Strategy]") -> str:  # type: ignore[name-defined]
    """Format strategy overview as a table (level 2)."""
    lines = ["\n[Strategies generated]",
             f"  {'ID':<4}  {'Name':<28}  {'Family':<20}  {'Predicted regime':<24}  {'Conf':>4}"]
    lines.append("  " + "─" * 85)
    for s in strategies:
        lines.append(
            f"  {s.id:<4}  {s.name[:28]:<28}  {s.family[:20]:<20}"
            f"  {s.predicted_winning_regime[:24]:<24}  {s.llm_confidence:>4.2f}"
        )
    return "\n".join(lines)


def _fmt_shape_results(results: "list[StrategyShapeResult]") -> str:  # type: ignore[name-defined]
    """Format per-shape profiling results as a compact table (level 2)."""
    lines = [f"  {'Shape':<10}  {'TFLOPS':>8}  {'vs baseline':>12}"]
    lines.append("  " + "─" * 34)
    for r in results:
        lines.append(
            f"  {r.shape_point.label:<10}  {r.achieved_tflops:>8.1f}  {r.vs_baseline_pct:>+11.1f}%"
        )
    return "\n".join(lines)


def _fmt_report(report: "ProfilingReport") -> str:  # type: ignore[name-defined]
    """Format a full ProfilingReport for debug output (level 3)."""
    r = report
    lines = [
        "  [ProfilingReport]",
        f"  Roofline:  {r.achieved_tflops:.1f} TFLOPS / peak {r.peak_tflops:.0f}"
        f"  |  arith intensity {r.arithmetic_intensity:.2f} FLOP/byte"
        f"  |  bottleneck: {r.bottleneck}  |  utilization {r.utilization_pct:.1%}",
        f"  Memory:    L1 {r.l1_hit_rate:.1%}  L2 {r.l2_hit_rate:.1%}"
        f"  |  DRAM {r.dram_bandwidth_gbs:.0f} GB/s (peak {r.peak_bandwidth_gbs:.0f})"
        f"  |  bank conflicts {r.shared_mem_bank_conflicts}",
        f"             global load eff {r.global_load_efficiency:.1%}"
        f"  |  store eff {r.global_store_efficiency:.1%}",
        f"  Warp:      efficiency {r.warp_efficiency:.1%}"
        f"  |  divergence {r.branch_divergence_pct:.1f}%"
        f"  |  occupancy {r.achieved_occupancy:.1%} (theoretical {r.theoretical_occupancy:.1%})"
        f"  |  replay overhead {r.instruction_replay_overhead:.1%}",
        f"  Instr:     tensor-core {r.tensor_core_utilization:.1%}"
        f"  |  FP16 {r.fp16_ops:,}  FP32 {r.fp32_ops:,}"
        f"  |  INT {r.int_ops:,}  special {r.special_func_ops:,}",
        f"  Latency:   {r.latency_ms:.3f} ms",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _extract_json(text: str) -> str:
    """Extract the first JSON object or array from text."""
    # Try to find JSON between ```json ... ``` markers first
    match = re.search(r"```json\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1)
    # Find the first { or [ and try to parse from there
    for start_char, end_char in (("{", "}"), ("[", "]")):
        idx = text.find(start_char)
        if idx >= 0:
            # Find matching closing bracket
            depth = 0
            for i, ch in enumerate(text[idx:], idx):
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                    if depth == 0:
                        return text[idx : i + 1]
    return text


def _extract_code_block(text: str) -> str:
    """Extract the first ```python ... ``` code block from text."""
    match = re.search(r"```python\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    # Fallback: ```...```
    match = re.search(r"```\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return ""


def _make_hardware_id(device_name: str) -> str:
    """Normalize device name to a slug, e.g. 'NVIDIA A100-SXM4-80GB' → 'a100-sxm4-80gb'."""
    return re.sub(r"[^a-z0-9]+", "-", device_name.lower()).strip("-")
