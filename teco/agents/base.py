"""Base agent interface for TECO agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from teco.orchestration.context import OptimizationContext


class BaseAgent(ABC):
    """Abstract base for all TECO agents.

    Each agent receives an OptimizationContext and returns an updated context.
    Agents are stateless — all state lives in OptimizationContext.
    """

    def __init__(self, model_id: str = "claude-opus-4-6") -> None:
        self.model_id = model_id

    @abstractmethod
    def run(self, context: OptimizationContext) -> OptimizationContext:
        """Execute this agent's task and return the updated context."""

    def _llm_call(self, system: str, user: str, **kwargs: Any) -> str:
        """Make a direct Anthropic API call with tool use disabled.

        Used for structured reasoning tasks (strategy planning, reflection).
        Returns the text content of the first text block.
        """
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=self.model_id,
            max_tokens=kwargs.get("max_tokens", 4096),
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
