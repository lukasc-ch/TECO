"""ReflectorAgent stub — Phase 2 implementation.

Responsibilities (future):
  - Runs after each completed optimization run
  - Reads the run record and all strategy shape results
  - Identifies patterns that repeat across >= 3 independent runs
  - Synthesizes new Lesson entries for knowledge/lessons/
  - Updates confidence scores in existing Pattern files
  - Produces a human-readable optimization report
"""

from __future__ import annotations

from teco.agents.base import BaseAgent
from teco.orchestration.context import OptimizationContext


class ReflectorAgent(BaseAgent):
    def run(self, context: OptimizationContext) -> OptimizationContext:
        # Stub: logs a brief summary but does not synthesize lessons yet
        context.log("[ReflectorAgent] Stub — run record written; lesson synthesis Phase 2", level=2)
        return context
