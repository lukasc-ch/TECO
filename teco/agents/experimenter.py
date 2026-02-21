"""ExperimenterAgent stub — Phase 2 implementation.

Responsibilities (future):
  - Triggered by OptimizerAgent when the bottleneck is ambiguous or ncu data is inconclusive
  - Designs and executes isolated microbenchmarks:
      - Block size sweeps
      - Occupancy experiments (register count vs. parallelism trade-off)
      - Memory access pattern tests (coalesced vs. strided vs. random)
      - Tensor core vs. scalar path comparisons
  - Returns partial conclusions that feed back into the optimizer loop
  - Self-reflects to extract generalizable findings for the knowledge store
"""

from __future__ import annotations

from teco.agents.base import BaseAgent
from teco.orchestration.context import OptimizationContext


class ExperimenterAgent(BaseAgent):
    def run(self, context: OptimizationContext) -> OptimizationContext:
        # Stub: no-op until Phase 2
        context.log("[ExperimenterAgent] Stub — skipping (Phase 2 not yet implemented)", level=2)
        return context
