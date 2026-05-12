"""pocket-news — fetch, deduplicate, and synthesize news with a local LLM."""

from __future__ import annotations
import logging

from .agent import NewsAgent
from .config import AgentConfig
from .exceptions import (
    ConfigurationError,
    FetcherError,
    OllamaUnavailableError,
    PocketNewsError,
    RateLimitError,
    SynthesisParseError,
)
from .models import Article, SourceCitation, SynthesizedArticle

__version__ = "0.1.0"

__all__ = [
    "NewsAgent",
    "AgentConfig",
    "PocketNewsError",
    "ConfigurationError",
    "FetcherError",
    "RateLimitError",
    "OllamaUnavailableError",
    "SynthesisParseError",
    "Article",
    "SourceCitation",
    "SynthesizedArticle",
    "configure_logging",
    "__version__",
]


def configure_logging(level: str = "INFO") -> None:
    """Configure pocket-news logging. Call once at application startup."""
    pkg = logging.getLogger("pocket_news")
    pkg.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not pkg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        pkg.addHandler(h)
