import os
import pytest
from pocket_news.config import AgentConfig
from pocket_news.exceptions import ConfigurationError


def test_defaults():
    config = AgentConfig()
    assert config.ollama_model == "qwen3:14b"
    assert config.max_articles_per_source == 5
    assert config.tone == "neutral"
    assert config.cache_enabled is True


def test_env_var_key_resolution(monkeypatch):
    monkeypatch.setenv("WORLD_NEWS_API_KEY", "env-key-123")
    config = AgentConfig()
    assert config.world_news_api_key == "env-key-123"


def test_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("WORLD_NEWS_API_KEY", "env-key")
    config = AgentConfig(world_news_api_key="kwarg-key")
    assert config.world_news_api_key == "kwarg-key"


def test_missing_key_is_none(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig()
    assert config.world_news_api_key is None


def test_max_articles_out_of_range():
    with pytest.raises(ConfigurationError):
        AgentConfig(max_articles_per_source=100)


def test_lookback_days_out_of_range():
    with pytest.raises(ConfigurationError):
        AgentConfig(lookback_days=0)


def test_invalid_tone():
    with pytest.raises(ConfigurationError):
        AgentConfig(tone="sarcastic")


def test_negative_cache_ttl():
    with pytest.raises(ConfigurationError):
        AgentConfig(cache_ttl_minutes=-1)


def test_resolved_fetchers_filters_none(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig(world_news_api_key="key-a", the_news_api_key=None, mediastack_api_key=None)
    result = config.resolved_fetchers()
    assert len(result) == 1
    assert result[0][0] == "world_news"


def test_resolved_fetchers_respects_enabled_list(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig(
        world_news_api_key="key-a",
        the_news_api_key="key-b",
        mediastack_api_key="key-c",
        enabled_fetchers=["world_news"],
    )
    result = config.resolved_fetchers()
    assert [fid for fid, _ in result] == ["world_news"]
