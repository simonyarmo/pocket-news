"""Core Pydantic v2 data models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Article(BaseModel):
    """A single news article returned by a fetcher."""

    fetcher: str
    source_id: str | None = None
    title: str
    url: str
    snippet: str | None = None
    content: str | None = None
    author: str | None = None
    outlet: str
    outlet_domain: str | None = None
    language: str | None = None
    country: str | None = None
    categories: list[str] = []
    published_at: datetime | None = None
    image_url: str | None = None


class SourceCitation(BaseModel):
    """Attribution record for one input article."""

    outlet: str
    title: str
    url: str
    published_at: datetime | None = None
    fetcher: str


class SynthesizedArticle(BaseModel):
    """Final output of NewsAgent.research()."""

    status: Literal["ok", "no_results", "partial"] = "ok"
    topic: str
    output_language: str

    headline: str
    lead: str
    body: str
    key_points: list[str]

    featured_image_url: str | None = None
    featured_image_b64: str | None = None
    featured_image_mime: str | None = None

    sources: list[SourceCitation] = []
    model: str
    generated_at: datetime
    article_count: int
    fetcher_status: dict[str, str]
