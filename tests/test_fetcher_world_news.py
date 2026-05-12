# tests/test_fetcher_world_news.py
import json
import os
from datetime import timezone
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.world_news import WorldNewsFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "world_news_response.json").read_text()
)
BASE_URL = "https://api.worldnewsapi.com/search-news"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = WorldNewsFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].fetcher == "world_news"
    assert articles[0].source_id == "123456789"
    assert articles[0].outlet == "Reuters"
    assert articles[0].content is not None  # World News returns full text
    assert articles[0].snippet == "The EU has started enforcing the AI Act."
    assert articles[0].author == "Jane Doe"
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo == timezone.utc
    assert articles[1].image_url is None
    assert articles[1].snippet is not None  # falls back to first 300 chars of text


@respx.mock
def test_401_raises_configuration_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(401))
    fetcher = WorldNewsFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_429_raises_rate_limit_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(429))
    fetcher = WorldNewsFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = WorldNewsFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("WORLD_NEWS_API_KEY"),
    reason="WORLD_NEWS_API_KEY not set",
)
def test_integration_live():
    fetcher = WorldNewsFetcher(api_key=os.environ["WORLD_NEWS_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "world_news" for a in articles)
