"""TECO configuration.

Settings are loaded in this priority order (highest wins):
  1. Environment variables  (TECO_* or ANTHROPIC_API_KEY)
  2. teco.toml in the current working directory
  3. ~/.config/teco/teco.toml
  4. Built-in defaults

Example teco.toml:
  [llm]
  model    = "claude-opus-4-6"
  api_key  = "sk-ant-..."   # or set ANTHROPIC_API_KEY
  base_url = "https://api.anthropic.com"  # override for proxies

  [output]
  verbosity = 1  # 0=quiet  1=normal  2=verbose  3=debug
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


_CONFIG_SEARCH_PATHS: list[Path] = [
    Path("teco.toml"),
    Path.home() / ".config" / "teco" / "teco.toml",
]


def _load_toml_full() -> dict:
    """Return the full parsed toml dict (all sections), or {} if no config file found."""
    for path in _CONFIG_SEARCH_PATHS:
        if path.exists():
            with path.open("rb") as f:
                return tomllib.load(f)
    return {}


_LLM_DEFAULTS: dict[str, str] = {
    "model":    "claude-opus-4-6",
    "api_key":  "",
    "base_url": "",
}


class LLMConfig:
    """Resolved LLM configuration."""

    def __init__(self) -> None:
        llm = {k: str(v) for k, v in _load_toml_full().get("llm", {}).items()}

        self.model: str = (
            os.environ.get("TECO_MODEL")
            or llm.get("model")
            or _LLM_DEFAULTS["model"]
        )
        self.api_key: str = (
            os.environ.get("TECO_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or llm.get("api_key")
            or _LLM_DEFAULTS["api_key"]
        )
        self.base_url: str = (
            os.environ.get("TECO_BASE_URL")
            or llm.get("base_url")
            or _LLM_DEFAULTS["base_url"]
        )

    def anthropic_client(self) -> "anthropic.Anthropic":  # type: ignore[name-defined]
        import anthropic

        kwargs: dict[str, str] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)


class OutputConfig:
    """Resolved output / verbosity configuration."""

    def __init__(self) -> None:
        out = _load_toml_full().get("output", {})
        self.verbosity: int = int(
            os.environ.get("TECO_VERBOSITY") or out.get("verbosity", 1)
        )


# Module-level singletons — loaded once per process.
llm_config = LLMConfig()
output_config = OutputConfig()
