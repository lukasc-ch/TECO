"""Web search and documentation fetching tool (stub — to be implemented with LearnerAgent)."""

from __future__ import annotations


def web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web for documentation, papers, or examples.

    Returns list of {title, url, snippet} dicts.
    Stub: raises NotImplementedError until LearnerAgent is implemented.
    """
    raise NotImplementedError(
        "web_search is not yet implemented. "
        "LearnerAgent (Phase 2) will implement this via an HTTP search API."
    )


def doc_fetch(url: str) -> str:
    """Fetch and extract text content from a documentation URL.

    Returns plain text content.
    Stub: raises NotImplementedError until LearnerAgent is implemented.
    """
    raise NotImplementedError(
        "doc_fetch is not yet implemented. "
        "LearnerAgent (Phase 2) will implement this via httpx + HTML stripping."
    )
