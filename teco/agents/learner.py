"""LearnerAgent stub — Phase 2 implementation.

Responsibilities (future):
  - Crawl official documentation for target language and hardware
  - Search arXiv and GitHub for relevant optimization papers and code
  - Summarize findings into knowledge/lessons/ files
  - Run before the optimizer loop when encountering a new language/hardware combination
"""

from __future__ import annotations

from teco.agents.base import BaseAgent
from teco.orchestration.context import OptimizationContext


class LearnerAgent(BaseAgent):
    def run(self, context: OptimizationContext) -> OptimizationContext:
        # Stub: no-op until Phase 2
        if context.verbose:
            print("[LearnerAgent] Stub — skipping (Phase 2 not yet implemented)")
        return context
