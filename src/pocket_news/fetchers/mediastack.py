"""MediaStack news fetcher."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "http://api.mediastack.com/v1/news"


class MediaStackFetcher(BaseFetcher):
    """Fetches articles from MediaStack API."""

    id: ClassVar[str] = "mediastack"
    display_name: ClassVar[str] = "MediaStack"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)
        date_range = f"{start.strftime('%Y-%m-%d')},{now.strftime('%Y-%m-%d')}"

        params = {
            "access_key": self.api_key,
            "keywords": topic,
            "languages": language,
            "date": date_range,
            "sort": "published_desc",
            "limit": max_articles,
        }

        try:
            response = self._client.get(BASE_URL, params=params)
        except Exception as exc:
            raise FetcherError(f"MediaStack request failed: {exc}") from exc

        if response.status_code >= 500:
            raise FetcherError(f"MediaStack server error: {response.status_code}")

        body = response.json()

        if "error" in body:
            code = body["error"].get("code", 0)
            message = body["error"].get("message", "Unknown error")
            if code == 101:
                raise ConfigurationError(f"MediaStack: {message}")
            if code == 104:
                raise RateLimitError(f"MediaStack: {message}")
            raise FetcherError(f"MediaStack error {code}: {message}")

        articles = []
        for item in body.get("data", []):
            outlet_name, domain = resolve_outlet(item.get("url", ""))
            source = item.get("source")
            if source:
                outlet_name = source

            published_at = None
            raw_date = item.get("published_at")
            if raw_date:
                try:
                    published_at = dateutil_parser.isoparse(raw_date).astimezone(timezone.utc)
                except Exception:
                    pass

            snippet = item.get("description")
            if snippet:
                snippet = _strip_html(snippet)

            articles.append(
                Article(
                    fetcher=self.id,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    author=item.get("author"),
                    outlet=outlet_name,
                    outlet_domain=domain,
                    language=item.get("language"),
                    country=item.get("country"),
                    categories=[item["category"]] if item.get("category") else [],
                    published_at=published_at,
                    image_url=item.get("image") or None,
                )
            )

        return articles
