# tests/test_cache.py
import time
import pytest
from datetime import datetime, timezone
from pocket_news.cache import CacheStore
from pocket_news.models import SynthesizedArticle


def test_set_and_get_roundtrip(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "EU AI Act", "en", "standard", "qwen3:14b")
    result = cache.get("EU AI Act", "en", "standard", "qwen3:14b")
    assert result is not None
    assert result.headline == sample_synthesized_article.headline


def test_cache_miss_returns_none(tmp_path):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    assert cache.get("unknown topic", "en", "standard", "qwen3:14b") is None


def test_stale_file_is_miss(tmp_path, sample_synthesized_article, monkeypatch):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=1)
    cache.set(sample_synthesized_article, "topic", "en", "standard", "qwen3:14b")
    orig = time.time
    monkeypatch.setattr(time, "time", lambda: orig() + 120)
    assert cache.get("topic", "en", "standard", "qwen3:14b") is None


def test_no_results_not_cached(tmp_path):
    no_results = SynthesizedArticle(
        status="no_results", topic="x", output_language="en",
        headline="", lead="", body="", key_points=[],
        model="qwen3:14b", generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=0, fetcher_status={},
    )
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(no_results, "x", "en", "standard", "qwen3:14b")
    assert cache.get("x", "en", "standard", "qwen3:14b") is None


def test_clear_returns_count(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "topic1", "en", "standard", "qwen3:14b")
    cache.set(sample_synthesized_article, "topic2", "en", "standard", "qwen3:14b")
    assert cache.clear() == 2
    assert list(tmp_path.glob("*.json")) == []


def test_key_includes_language_and_length(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "topic", "en", "standard", "qwen3:14b")
    assert cache.get("topic", "es", "standard", "qwen3:14b") is None
    assert cache.get("topic", "en", "brief", "qwen3:14b") is None
