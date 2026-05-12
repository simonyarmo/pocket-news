# tests/test_fetcher_mediastack.py
import json
import os
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.mediastack import MediaStackFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "mediastack_response.json").read_text()
)
BASE_URL = "http://api.mediastack.com/v1/news"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = MediaStackFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].outlet == "BBC News"
    assert articles[0].fetcher == "mediastack"
    assert articles[0].author == "John Smith"
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo is not None
    assert articles[1].image_url is None


@respx.mock
def test_mediastack_error_in_body():
    error_body = {"error": {"code": 104, "message": "User has reached their monthly request limit."}}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=error_body))
    fetcher = MediaStackFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_invalid_key_error():
    error_body = {"error": {"code": 101, "message": "Invalid API key."}}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=error_body))
    fetcher = MediaStackFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_http_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = MediaStackFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("MEDIASTACK_API_KEY"),
    reason="MEDIASTACK_API_KEY not set",
)
def test_integration_live():
    fetcher = MediaStackFetcher(api_key=os.environ["MEDIASTACK_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "mediastack" for a in articles)
