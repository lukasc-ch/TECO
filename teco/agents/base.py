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


def _is_retryable(exc: Exception) -> bool:
    """Return True if *exc* is a transient OpenAI API error worth retrying."""
    try:
        import openai
    except ImportError:
        return False
    return isinstance(
        exc,
        (
            openai.RateLimitError,
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.InternalServerError,
        ),
    )


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
        """Make an OpenAI-compatible chat completion API call.

        Used for structured reasoning tasks (strategy planning, reflection).
        Returns the text content of the assistant's reply.

        Retries transient errors (rate-limit, timeout, connection, server)
        with exponential backoff, reporting each attempt to stderr so the
        user can see what is happening.
        """
        client = self._config.openai_client()
        max_tokens = kwargs.get("max_tokens", 4096)

        backoff = _INITIAL_BACKOFF_S
        last_error_msg = ""

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                # Success — extract text content
                content = response.choices[0].message.content
                return content if content is not None else ""

            except Exception as exc:
                last_error_msg = f"{type(exc).__name__}: {exc}"
                retryable = _is_retryable(exc)
                print(
                    f"[LLM] Attempt {attempt}/{_MAX_RETRIES} failed — {last_error_msg}",
                    file=sys.stderr,
                )
                if retryable and attempt < _MAX_RETRIES:
                    print(f"[LLM] Retrying in {backoff:.1f}s …", file=sys.stderr)
                    time.sleep(backoff)
                    backoff = min(backoff * _BACKOFF_MULTIPLIER, _MAX_BACKOFF_S)
                    continue

                if not retryable:
                    raise LLMError(
                        f"LLM API returned an error after {attempt} attempt(s): {last_error_msg}\n"
                        f"  model={self.model_id}  base_url={self._config.base_url or '(default)'}"
                    ) from exc

        # All retries exhausted (only reached if every attempt hit a retryable error)
        raise LLMError(
            f"LLM API call failed after {_MAX_RETRIES} attempts: {last_error_msg}\n"
            f"  model={self.model_id}  base_url={self._config.base_url or '(default)'}"
        )
