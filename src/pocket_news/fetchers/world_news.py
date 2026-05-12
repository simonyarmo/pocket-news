"""World News API fetcher (worldnewsapi.com)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldnewsapi.com/search-news"


class WorldNewsFetcher(BaseFetcher):
    """Fetches articles from World News API."""

    id: ClassVar[str] = "world_news"
    display_name: ClassVar[str] = "World News API"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        earliest = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        params = {
            "text": topic,
            "language": language,
            "earliest-publish-date": earliest.strftime("%Y-%m-%dT%H:%M:%S"),
            "sort": "publish-time",
            "sort-direction": "DESC",
            "number": max_articles,
        }
        headers = {"x-api-key": self.api_key}

        try:
            response = self._client.get(BASE_URL, params=params, headers=headers)
        except Exception as exc:
            raise FetcherError(f"World News API request failed: {exc}") from exc

        if response.status_code == 401:
            raise ConfigurationError("World News API: invalid API key")
        if response.status_code in (402, 429):
            raise RateLimitError("World News API: rate limit or quota exceeded")
        if response.status_code >= 500:
            raise FetcherError(f"World News API server error: {response.status_code}")

        body = response.json()
        articles = []

        for item in body.get("news", []):
            outlet_name, domain = resolve_outlet(item.get("url", ""))

            raw_text = item.get("text", "")
            content = _strip_html(raw_text) if raw_text else None

            raw_summary = item.get("summary")
            if raw_summary:
                snippet = _strip_html(raw_summary)
            elif content:
                snippet = content[:300]
            else:
                snippet = None

            published_at = None
            raw_date = item.get("publish_date")
            if raw_date:
                try:
                    published_at = dateutil_parser.parse(raw_date).replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            articles.append(
                Article(
                    fetcher=self.id,
                    source_id=str(item["id"]) if item.get("id") is not None else None,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    content=content,
                    author=item.get("author") or None,
                    outlet=outlet_name,
                    outlet_domain=domain,
                    language=item.get("language"),
                    country=item.get("source_country"),
                    categories=item.get("categories") or [],
                    published_at=published_at,
                    image_url=item.get("image") or None,
                )
            )

        return articles
