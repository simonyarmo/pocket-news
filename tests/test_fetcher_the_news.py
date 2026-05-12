# tests/test_fetcher_the_news.py
import json
import os
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.the_news import TheNewsAPIFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "the_news_response.json").read_text()
)
BASE_URL = "https://api.thenewsapi.com/v1/news/all"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].fetcher == "the_news"
    assert articles[0].source_id == "6b100cae-f689-4b5e-8a7d-1234567890ab"
    assert articles[0].outlet == "Politico"
    assert articles[0].snippet is not None
    assert articles[1].image_url is None
    assert articles[1].country == "gb"


@respx.mock
def test_snippet_falls_back_to_snippet_field():
    import copy
    fixture = copy.deepcopy(FIXTURE)
    fixture["data"][0]["description"] = None
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=fixture))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert articles[0].snippet == "Businesses must comply by August 2026."


@respx.mock
def test_401_raises_configuration_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(401))
    fetcher = TheNewsAPIFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_429_raises_rate_limit_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(429))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("THE_NEWS_API_KEY"),
    reason="THE_NEWS_API_KEY not set",
)
def test_integration_live():
    fetcher = TheNewsAPIFetcher(api_key=os.environ["THE_NEWS_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "the_news" for a in articles)
