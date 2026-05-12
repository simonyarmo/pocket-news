# tests/test_agent.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from pocket_news.agent import NewsAgent
from pocket_news.config import AgentConfig
from pocket_news.models import Article


def _config(**kw):
    defaults = dict(
        world_news_api_key="k1",
        the_news_api_key=None,
        mediastack_api_key=None,
        cache_enabled=False,
        fetch_image_data=False,
    )
    defaults.update(kw)
    return AgentConfig(**defaults)


def _article(fetcher="world_news"):
    return Article(
        fetcher=fetcher, title="Title", url=f"https://{fetcher}.com",
        outlet="Example", snippet="snippet.",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )


PARSED = {"headline": "Headline", "lead": "Lead.", "body": "Body.", "key_points": ["P1"]}


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_research_returns_synthesized_article(MockReg, MockSynth):
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = [_article()]
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)
    MockSynth.return_value.synthesize.return_value = PARSED

    agent = NewsAgent(config=_config())
    result = agent.research("EU AI Act")
    assert result.headline == "Headline"
    assert result.topic == "EU AI Act"
    assert result.output_language == "en"
    assert result.status in ("ok", "partial")


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_all_empty_returns_no_results(MockReg, MockSynth):
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = []
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)

    agent = NewsAgent(config=_config())
    result = agent.research("obscure topic")
    assert result.status == "no_results"
    assert result.headline == ""
    MockSynth.return_value.synthesize.assert_not_called()


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_cache_hit_skips_fetch(MockReg, MockSynth, tmp_path, sample_synthesized_article):
    from pocket_news.cache import CacheStore
    store = CacheStore(tmp_path, 60)
    store.set(sample_synthesized_article, "EU AI Act enforcement", "en", "standard", "qwen3:14b")

    mock_fetcher = MagicMock()
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)

    config = AgentConfig(
        world_news_api_key="k1", the_news_api_key=None, mediastack_api_key=None,
        cache_enabled=True, cache_dir=tmp_path, cache_ttl_minutes=60, fetch_image_data=False,
    )
    agent = NewsAgent(config=config)
    result = agent.research("EU AI Act enforcement")
    assert result.headline == sample_synthesized_article.headline
    mock_fetcher.fetch.assert_not_called()
    MockSynth.return_value.synthesize.assert_not_called()
