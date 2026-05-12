"""NewsAgent — orchestrates fetch, dedupe, synthesize, cache."""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cache import CacheStore
from .config import AgentConfig
from .dedupe import ArticleDeduplicator
from .exceptions import FetcherError, PocketNewsError
from .fetchers import FETCHER_REGISTRY
from .images import fetch_and_encode_image
from .languages import normalize_language
from .models import Article, SourceCitation, SynthesizedArticle
from .prompts import LENGTH_TO_WORD_COUNT
from .synthesizer import OllamaSynthesizer

logger = logging.getLogger(__name__)


class NewsAgent:
    """Fetches, deduplicates, and synthesizes news for a given topic."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self._config = config or AgentConfig()
        self._synthesizer: Optional[OllamaSynthesizer] = None
        self._cache: Optional[CacheStore] = None

    def _get_synthesizer(self) -> OllamaSynthesizer:
        if self._synthesizer is None:
            self._synthesizer = OllamaSynthesizer(
                model=self._config.ollama_model,
                host=self._config.ollama_host,
                timeout_seconds=self._config.ollama_timeout_seconds,
                disable_thinking_mode=self._config.disable_thinking_mode,
            )
        return self._synthesizer

    def _get_cache(self) -> Optional[CacheStore]:
        if not self._config.cache_enabled:
            return None
        if self._cache is None:
            cache_dir = Path(self._config.cache_dir) if self._config.cache_dir else None
            self._cache = CacheStore(cache_dir, self._config.cache_ttl_minutes)
        return self._cache

    def _fetch_all(self, topic: str) -> tuple[list[Article], dict[str, str]]:
        fetcher_pairs = self._config.resolved_fetchers()
        fetcher_status: dict[str, str] = {
            fid: "disabled"
            for fid in FETCHER_REGISTRY
            if not any(fid == p[0] for p in fetcher_pairs)
        }
        if not fetcher_pairs:
            return [], fetcher_status

        def _fetch_one(fetcher_id: str, api_key: str) -> tuple[str, list[Article], str]:
            fetcher_cls = FETCHER_REGISTRY[fetcher_id]
            fetcher = fetcher_cls(api_key, timeout=self._config.fetch_timeout_seconds)
            try:
                arts = fetcher.fetch(
                    topic, self._config.max_articles_per_source,
                    "en", self._config.lookback_days,
                )
                return fetcher_id, arts, "ok" if arts else "empty"
            except FetcherError as exc:
                logger.warning("Fetcher %s failed: %s", fetcher_id, exc)
                return fetcher_id, [], f"failed: {exc}"
            finally:
                fetcher.close()

        all_articles: list[Article] = []
        with ThreadPoolExecutor(max_workers=len(fetcher_pairs)) as pool:
            futures = [pool.submit(_fetch_one, fid, key) for fid, key in fetcher_pairs]
            for fid, arts, status in [f.result() for f in as_completed(futures)]:
                fetcher_status[fid] = status
                all_articles.extend(arts)

        return all_articles, fetcher_status

    def research(
        self,
        topic: str,
        language: str = "en",
        length: str = "standard",
    ) -> SynthesizedArticle:
        """Fetch, deduplicate, and synthesize articles for a topic."""
        iso_code, display_name = normalize_language(language)
        word_count = LENGTH_TO_WORD_COUNT.get(length, LENGTH_TO_WORD_COUNT["standard"])

        cache = self._get_cache()
        if cache:
            cached = cache.get(topic, iso_code, length, self._config.ollama_model)
            if cached:
                return cached

        raw_articles, fetcher_status = self._fetch_all(topic)

        if not raw_articles:
            return SynthesizedArticle(
                status="no_results", topic=topic, output_language=iso_code,
                headline="", lead="", body="", key_points=[],
                model=self._config.ollama_model,
                generated_at=datetime.now(timezone.utc),
                article_count=0, fetcher_status=fetcher_status,
            )

        articles = ArticleDeduplicator.deduplicate(raw_articles)

        featured_image_url = featured_image_b64 = featured_image_mime = None
        for art in articles:
            if art.image_url:
                featured_image_url = art.image_url
                if self._config.fetch_image_data:
                    featured_image_b64, featured_image_mime = fetch_and_encode_image(
                        art.image_url, self._config.image_max_bytes, self._config.image_fetch_timeout
                    )
                break

        parsed = self._get_synthesizer().synthesize(
            topic, articles, display_name, word_count, self._config.tone
        )
        status = "partial" if any(s.startswith("failed:") for s in fetcher_status.values()) else "ok"

        result = SynthesizedArticle(
            status=status, topic=topic, output_language=iso_code,
            headline=parsed["headline"], lead=parsed["lead"], body=parsed["body"],
            key_points=parsed["key_points"],
            featured_image_url=featured_image_url, featured_image_b64=featured_image_b64,
            featured_image_mime=featured_image_mime,
            sources=[
                SourceCitation(outlet=a.outlet, title=a.title, url=a.url,
                               published_at=a.published_at, fetcher=a.fetcher)
                for a in articles
            ],
            model=self._config.ollama_model,
            generated_at=datetime.now(timezone.utc),
            article_count=len(articles), fetcher_status=fetcher_status,
        )
        if cache:
            cache.set(result, topic, iso_code, length, self._config.ollama_model)
        return result

    def research_batch(
        self,
        topics: list[str],
        language: str = "en",
        length: str = "standard",
        max_workers: int = 3,
    ) -> list[SynthesizedArticle]:
        """Runs research() for each topic concurrently. Preserves input order."""
        results: dict[int, SynthesizedArticle] = {}

        def _do(idx: int, topic: str) -> tuple[int, SynthesizedArticle]:
            try:
                return idx, self.research(topic, language=language, length=length)
            except PocketNewsError as exc:
                logger.warning("research() failed for %r: %s", topic, exc)
                iso_code, _ = normalize_language(language)
                return idx, SynthesizedArticle(
                    status="no_results", topic=topic, output_language=iso_code,
                    headline="", lead="", body="", key_points=[],
                    model=self._config.ollama_model,
                    generated_at=datetime.now(timezone.utc),
                    article_count=0, fetcher_status={},
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_do, i, t) for i, t in enumerate(topics)]
            for idx, article in [f.result() for f in as_completed(futures)]:
                results[idx] = article

        return [results[i] for i in range(len(topics))]

    def clear_cache(self) -> int:
        """Clears the local cache. Returns number of entries removed."""
        cache = self._get_cache()
        return cache.clear() if cache else 0
