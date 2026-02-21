"""Base agent interface for TECO agents."""

from __future__ import annotations

import sys
import time
from abc import ABC, abstractmethod
from typing import Any

from teco.config import LLMConfig, llm_config
from teco.orchestration.context import OptimizationContext

# Retry configuration for transient API errors
_MAX_RETRIES = 5
_INITIAL_BACKOFF_S = 1.0
_BACKOFF_MULTIPLIER = 2.0
_MAX_BACKOFF_S = 60.0

# Error types that are considered transient and worth retrying
_RETRYABLE_ERRORS = {"api_error", "overloaded", "rate_limit_error", "timeout"}


class LLMError(RuntimeError):
    """Raised when the LLM API returns an unrecoverable error."""


class BaseAgent(ABC):
    """Abstract base for all TECO agents.

    Each agent receives an OptimizationContext and returns an updated context.
    Agents are stateless — all state lives in OptimizationContext.
    """

    def __init__(self, model_id: str | None = None, config: LLMConfig | None = None) -> None:
        self._config = config or llm_config
        # model_id argument overrides config (e.g. when passed via CLI --model flag)
        self.model_id = model_id or self._config.model

    @abstractmethod
    def run(self, context: OptimizationContext) -> OptimizationContext:
        """Execute this agent's task and return the updated context."""

    def _llm_call(self, system: str, user: str, **kwargs: Any) -> str:
        """Make a direct Anthropic API call with tool use disabled.

        Used for structured reasoning tasks (strategy planning, reflection).
        Returns the text content of the first text block.

        Retries transient errors (overloaded, rate-limit, timeout) with
        exponential backoff, reporting each attempt to stderr so the user
        can see what is happening.
        """
        client = self._config.anthropic_client()
        max_tokens = kwargs.get("max_tokens", 4096)

        backoff = _INITIAL_BACKOFF_S
        last_error_msg = ""

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.messages.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            except Exception as exc:
                # Network-level or SDK-level exceptions (connection reset, timeout, etc.)
                last_error_msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"[LLM] Attempt {attempt}/{_MAX_RETRIES} failed — {last_error_msg}",
                    file=sys.stderr,
                )
                if attempt < _MAX_RETRIES:
                    print(f"[LLM] Retrying in {backoff:.1f}s …", file=sys.stderr)
                    time.sleep(backoff)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S)
                continue

            # The Anthropic SDK may parse an error response from a proxy
            # (e.g. OpenRouter) into a Message with type='error' and content=None.
            error_info = getattr(response, "error", None)
            if response.content is None or getattr(response, "type", None) == "error":
                error_type = error_info.get("type", "unknown") if isinstance(error_info, dict) else "unknown"
                error_message = error_info.get("message", "unknown error") if isinstance(error_info, dict) else str(error_info)
                last_error_msg = f"{error_type}: {error_message}"

                is_retryable = error_type.lower() in _RETRYABLE_ERRORS or "overload" in error_message.lower()
                print(
                    f"[LLM] Attempt {attempt}/{_MAX_RETRIES} — API error: {last_error_msg}",
                    file=sys.stderr,
                )
                if is_retryable and attempt < _MAX_RETRIES:
                    print(f"[LLM] Retrying in {backoff:.1f}s …", file=sys.stderr)
                    time.sleep(backoff)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S)
                    continue

                # Non-retryable or exhausted retries
                raise LLMError(
                    f"LLM API returned an error after {attempt} attempt(s): {last_error_msg}\n"
                    f"  model={self.model_id}  base_url={self._config.base_url or '(default)'}"
                )

            # Success — extract text content
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        # All retries exhausted (only reached if every attempt hit the except branch)
        raise LLMError(
            f"LLM API call failed after {_MAX_RETRIES} attempts: {last_error_msg}\n"
            f"  model={self.model_id}  base_url={self._config.base_url or '(default)'}"
        )
