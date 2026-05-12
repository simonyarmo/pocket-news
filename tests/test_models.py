"""Tests for pocket_news.models."""

import pytest
from datetime import datetime, timezone
from pocket_news.models import Article, SourceCitation, SynthesizedArticle


def test_article_minimal():
    a = Article(fetcher="world_news", title="Test", url="https://example.com", outlet="Example")
    assert a.content is None
    assert a.categories == []


def test_article_json_roundtrip():
    a = Article(
        fetcher="the_news",
        title="Test Article",
        url="https://example.com/article",
        outlet="Example News",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )
    json_str = a.model_dump_json()
    restored = Article.model_validate_json(json_str)
    assert restored.title == a.title
    assert restored.published_at == a.published_at


def test_synthesized_article_defaults():
    art = SynthesizedArticle(
        topic="Test",
        output_language="en",
        headline="Headline",
        lead="Lead paragraph.",
        body="Body text.",
        key_points=["Point 1"],
        model="qwen3:14b",
        generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=1,
        fetcher_status={"world_news": "ok"},
    )
    assert art.status == "ok"
    assert art.featured_image_b64 is None
    assert art.sources == []


def test_synthesized_article_json_roundtrip(sample_synthesized_article):
    json_str = sample_synthesized_article.model_dump_json()
    restored = SynthesizedArticle.model_validate_json(json_str)
    assert restored.headline == sample_synthesized_article.headline
    assert len(restored.sources) == 3
    assert restored.status == "ok"


def test_no_results_status():
    art = SynthesizedArticle(
        status="no_results",
        topic="obscure topic",
        output_language="en",
        headline="",
        lead="",
        body="",
        key_points=[],
        model="qwen3:14b",
        generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=0,
        fetcher_status={"world_news": "empty", "the_news": "empty", "mediastack": "empty"},
    )
    assert art.status == "no_results"
