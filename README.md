# TECO -- The Experienced Code Optimizer

TECO is an agentic GPU kernel optimizer that combines LLM reasoning with real profiling feedback to maximize performance of GPU code — starting with Triton, with planned support for CUDA, TileLang, and AscendC.

Its distinguishing feature is **learned experience**: TECO behaves like a skilled student. Before optimizing, it studies documentation and known approaches. During optimization, it forms hypotheses, runs profiling experiments, and — when a bottleneck is ambiguous — decomposes the problem into isolated microbenchmarks to reach partial conclusions. Afterward, it reflects to extract generalizable lessons that inform future runs.

## How it works

**1. Strategy planning**
Rather than making incremental tweaks, TECO starts by generating several fundamentally different, mutually incompatible optimization strategies (e.g., tensor core rewrite vs. vectorized loads vs. shared memory tiling). Each strategy addresses the problem from a different angle.

**2. Shape-aware beam search**
Performance often depends on input shape — a strategy that wins for large matrices may lose for small ones. TECO profiles across a representative shape sweep (small / medium / large) and tracks which strategy wins each shape regime. Multiple strategies can co-exist as regime winners.

**3. Prune and deepen**
Each iteration, TECO re-scores all active strategies by LLM confidence given accumulated profiling data, prunes dominated branches, and deepens the best remaining ones. It narrows from N strategies to a final set of regime winners.

**4. Dispatch wrapper**
When multiple strategies win different shape regimes, TECO synthesizes a runtime dispatch wrapper that selects the optimal implementation based on the actual input dimensions.

**5. Knowledge accumulation**
Every optimization run is logged: strategies tried, profiling results across shapes, winning regimes, and prune reasons — including for losing strategies. This builds a structured knowledge base that future runs query to skip known dead ends and prioritize strategies that have worked before on similar kernels and hardware.

## Project structure

```
teco/
  agents/          # LLM-powered agents (OptimizerAgent is the core MVP)
  tools/           # GPU profiling (ncu), compilation, code editing
  knowledge/       # Schema + KnowledgeStore
  orchestration/   # OptimizationContext + main loop
  cli.py           # CLI entry point
knowledge/
  runs/            # One JSON per optimization run (auto-generated)
  patterns/        # YAML+Markdown per (strategy_family, hardware) pair
  lessons/         # Cross-cutting synthesized insights
TritonBench/       # Benchmark dataset (git submodule)
PLAN.md            # Full architecture and design document
```

## Getting started

**Requirements:**
- Python >= 3.11
- CUDA-capable NVIDIA GPU
- `ncu` (Nsight Compute) on `PATH`
- `ANTHROPIC_API_KEY` set in your environment

```bash
uv sync                                                                           # install dependencies
uv run pytest tests/test_knowledge_store.py tests/test_compiler.py                # test compiles & knowledge store (no GPU needed)
uv run teco optimize <kernel.py> --verbose                                        # optimize specific kernel


uv run teco optimize path/to/kernel.py --verbose                                  # optimize a kernel
uv run teco optimize TritonBench/data/TritonBench_G_v1/lightning_attention.py     # TritonBench example
uv run teco query a100-sxm4-80gb --tags attention,softmax                         # query knowledge base

 uv run teco optimize TritonBench/data/TritonBench_G_v1/max_reduction.py --verbosity 3 # output all info


uv run pytest                                                                     # full test suite (GPU)
```

## Roadmap

- **MVP (current):** OptimizerAgent — profile → strategy planning → beam search → dispatch wrapper → knowledge logging
- **Phase 2:** LearnerAgent (crawl docs/arXiv before optimization), ExperimenterAgent (isolated microbenchmarks for ambiguous bottlenecks), ReflectorAgent (LLM-powered lesson synthesis)
- **Phase 3:** Multi-language support (CUDA, TileLang, AscendC), translation between languages, remote GPU execution
