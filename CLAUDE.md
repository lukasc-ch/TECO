# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**TECO -- The Experienced Code Optimizer** is an agentic GPU kernel optimizer that improves
performance of GPU code (starting with Triton) by combining LLM reasoning with real profiling
feedback. Its distinguishing feature is learned experience: it behaves like a skilled student —
studying documentation, running targeted experiments, and reflecting to extract generalizable
lessons after each optimization run.

## Toolchain

- **Package manager**: `uv` (`uv sync`, `uv run <cmd>`)
- **Testing**: pytest (`uv run pytest`)
- **Linting/formatting**: Ruff (`uv run ruff check .`, `uv run ruff format .`)
- **Type checking**: mypy (`uv run mypy .`)
- **Notebooks**: Marimo

## System requirements

- CUDA-capable GPU (NVIDIA)
- `ncu` (Nsight Compute) on PATH — required for Stage 2 deep profiling
- Triton >= 3.0, PyTorch >= 2.0

## Project structure

```
teco/
  agents/          # LLM-powered agents (OptimizerAgent is the core MVP)
  tools/           # GPU profiling (ncu), compilation, code editing, search
  knowledge/       # Schema + KnowledgeStore (runs/, patterns/, lessons/)
  orchestration/   # OptimizationContext + main loop
  cli.py           # typer CLI: `teco optimize`, `teco query`, `teco rebuild-index`
knowledge/
  runs/            # One JSON per optimization run (auto-generated)
  patterns/        # YAML+Markdown per (strategy_family, hardware) pair
  lessons/         # Cross-cutting synthesized insights
  index.json       # Lightweight metadata cache (auto-maintained)
tests/
TritonBench/       # Git submodule — benchmark dataset for evaluation
PLAN.md            # Architecture and implementation plan
```

## Common commands

```bash
uv run teco optimize TritonBench/data/TritonBench_G_v1/lightning_attention.py --verbose
uv run teco query a100-sxm4-80gb --tags attention,softmax
uv run teco rebuild-index
uv run pytest
uv run ruff check .
uv run mypy teco/
```

## Key design decisions

- **Agent framework**: `smolagents` (Hugging Face) + OpenAI-compatible API (works with any provider via `base_url`)
- **Knowledge store**: hierarchical Markdown + JSON files (no database)
  - `patterns/`: machine-maintained YAML front-matter + human-editable Markdown body
  - Query via 3-pass funnel: hard filter → composite score → rank
- **Profiling**: two-stage — Stage 1 (`triton.testing.do_bench`, every iteration) +
  Stage 2 (`ncu --set full`, baseline + final per regime winner)
- **Strategy tree**: beam-search style — generate N incompatible strategies, prune/deepen
  each iteration by LLM confidence; multiple strategies can win for different shape regimes
- **Output**: dispatch wrapper that selects the optimal strategy at runtime based on input shape

## Development notes

- All agents are stateless — state lives in `OptimizationContext`
- `KernelCharacteristics` uses weighted probability vectors for fuzzy similarity matching
- Pattern files: YAML front-matter is auto-updated; Markdown body is never overwritten
- `index.json` is written atomically (write-then-rename) to prevent corruption
- `LearnerAgent`, `ExperimenterAgent`, `ReflectorAgent` are stubs — full implementation is Phase 2
