"""AgentConfig — configuration with API key resolution chain."""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .exceptions import ConfigurationError


def _resolve_key(name: str) -> str | None:
    val = os.getenv(name)
    if val:
        return val
    try:
        from . import _secrets  # type: ignore[import]

        val = getattr(_secrets, name, None)
        if val:
            return val
    except ImportError:
        pass
    return None


class AgentConfig(BaseModel):
    """All configuration knobs for NewsAgent. Every field has a sensible default."""

    world_news_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("WORLD_NEWS_API_KEY")
    )
    the_news_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("THE_NEWS_API_KEY")
    )
    mediastack_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("MEDIASTACK_API_KEY")
    )

    enabled_fetchers: list[str] | None = None
    max_articles_per_source: int = 5
    lookback_days: int = 7
    fetch_timeout_seconds: float = 15.0

    fetch_image_data: bool = True
    image_max_bytes: int = 2_000_000
    image_fetch_timeout: float = 5.0

    cache_enabled: bool = True
    cache_ttl_minutes: int = 60
    cache_dir: Any = None  # Path | None

    ollama_model: str = "qwen3:14b"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 120
    disable_thinking_mode: bool = True

    tone: str = "neutral"

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AgentConfig":
        if not 1 <= self.max_articles_per_source <= 50:
            raise ConfigurationError("max_articles_per_source must be between 1 and 50")
        if not 1 <= self.lookback_days <= 30:
            raise ConfigurationError("lookback_days must be between 1 and 30")
        if self.tone not in {"neutral", "analytical", "brief"}:
            raise ConfigurationError("tone must be 'neutral', 'analytical', or 'brief'")
        if self.cache_ttl_minutes < 0:
            raise ConfigurationError("cache_ttl_minutes must be >= 0")
        if self.image_max_bytes <= 0:
            raise ConfigurationError("image_max_bytes must be > 0")
        return self

    def resolved_fetchers(self) -> list[tuple[str, str]]:
        """Returns (fetcher_id, api_key) pairs for fetchers that are enabled and have keys."""
        candidates = [
            ("world_news", self.world_news_api_key),
            ("the_news", self.the_news_api_key),
            ("mediastack", self.mediastack_api_key),
        ]
        return [
            (fid, key)
            for fid, key in candidates
            if key and (self.enabled_fetchers is None or fid in self.enabled_fetchers)
        ]
