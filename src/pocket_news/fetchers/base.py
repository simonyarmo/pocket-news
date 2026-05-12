"""BaseFetcher abstract class and shared outlet utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.parse import urlparse

import httpx

from ..models import Article

DOMAIN_TO_OUTLET: dict[str, str] = {
    "bbc.co.uk": "BBC News",
    "bbc.com": "BBC News",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "theguardian.com": "The Guardian",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "npr.org": "NPR",
    "cnn.com": "CNN",
    "foxnews.com": "Fox News",
    "wsj.com": "The Wall Street Journal",
    "bloomberg.com": "Bloomberg",
    "ft.com": "Financial Times",
    "aljazeera.com": "Al Jazeera",
    "dw.com": "Deutsche Welle",
    "france24.com": "France 24",
    "economist.com": "The Economist",
    "politico.com": "Politico",
    "axios.com": "Axios",
    "theverge.com": "The Verge",
    "techcrunch.com": "TechCrunch",
    "arstechnica.com": "Ars Technica",
    "nypost.com": "New York Post",
    "nbcnews.com": "NBC News",
    "cbsnews.com": "CBS News",
    "abcnews.go.com": "ABC News",
}

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")


def _strip_html(text: str) -> str:
    text = _HTML_TAG.sub("", text)
    text = _HTML_ENTITY.sub(" ", text)
    return text.strip()


def resolve_outlet(url_or_domain: str) -> tuple[str, str]:
    """Returns (outlet_name, domain). Falls back to titlecased hostname segment."""
    raw = url_or_domain if "://" in url_or_domain else "https://" + url_or_domain
    try:
        parsed = urlparse(raw)
        domain = parsed.netloc.lower()
    except Exception:
        domain = url_or_domain.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    if domain in DOMAIN_TO_OUTLET:
        return DOMAIN_TO_OUTLET[domain], domain

    parts = domain.split(".")
    for i in range(1, len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in DOMAIN_TO_OUTLET:
            return DOMAIN_TO_OUTLET[candidate], domain

    base = parts[0].replace("-", " ").title() if parts else domain.title()
    return base, domain


class BaseFetcher(ABC):
    """Abstract base for all news API fetchers."""

    id: ClassVar[str]
    display_name: ClassVar[str]

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @abstractmethod
    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        """Fetch articles for topic. Raises FetcherError or RateLimitError on failure."""
        ...

    def close(self) -> None:
        self._client.close()
