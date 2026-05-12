"""The News API fetcher (thenewsapi.com)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thenewsapi.com/v1/news/all"
MAX_PAGE_REQUESTS = 3


class TheNewsAPIFetcher(BaseFetcher):
    """Fetches articles from The News API."""

    id: ClassVar[str] = "the_news"
    display_name: ClassVar[str] = "The News API"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        articles: list[Article] = []
        page = 1

        while len(articles) < max_articles and page <= MAX_PAGE_REQUESTS:
            params = {
                "api_token": self.api_key,
                "search": topic,
                "language": language,
                "published_after": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "sort": "published_at",
                "limit": min(max_articles - len(articles), 3),
                "page": page,
            }

            try:
                response = self._client.get(BASE_URL, params=params)
            except Exception as exc:
                raise FetcherError(f"The News API request failed: {exc}") from exc

            if response.status_code == 401:
                raise ConfigurationError("The News API: invalid API key")
            if response.status_code in (402, 429):
                raise RateLimitError("The News API: rate limit or quota exceeded")
            if response.status_code >= 500:
                raise FetcherError(f"The News API server error: {response.status_code}")

            body = response.json()
            data = body.get("data", [])
            if not data:
                break

            for item in data:
                snippet = item.get("description") or item.get("snippet")
                if snippet:
                    snippet = _strip_html(snippet)

                outlet_name, domain = resolve_outlet(item.get("source", ""))

                published_at = None
                raw_date = item.get("published_at")
                if raw_date:
                    try:
                        published_at = dateutil_parser.isoparse(raw_date).astimezone(timezone.utc)
                    except Exception:
                        pass

                articles.append(
                    Article(
                        fetcher=self.id,
                        source_id=item.get("uuid"),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=snippet,
                        outlet=outlet_name,
                        outlet_domain=domain,
                        language=item.get("language"),
                        country=item.get("locale"),
                        categories=item.get("categories") or [],
                        published_at=published_at,
                        image_url=item.get("image_url") or None,
                    )
                )

            meta = body.get("meta", {})
            if len(articles) >= meta.get("found", 0):
                break
            page += 1

        return articles
