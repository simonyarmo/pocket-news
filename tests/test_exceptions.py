"""Test exception hierarchy for pocket-news."""
import pytest
from pocket_news.exceptions import (
    PocketNewsError,
    ConfigurationError,
    FetcherError,
    RateLimitError,
    OllamaUnavailableError,
    SynthesisParseError,
)


def test_hierarchy():
    assert issubclass(ConfigurationError, PocketNewsError)
    assert issubclass(FetcherError, PocketNewsError)
    assert issubclass(RateLimitError, FetcherError)
    assert issubclass(OllamaUnavailableError, PocketNewsError)
    assert issubclass(SynthesisParseError, PocketNewsError)


def test_catch_as_base():
    with pytest.raises(PocketNewsError):
        raise OllamaUnavailableError("Ollama not running")

    with pytest.raises(PocketNewsError):
        raise RateLimitError("Rate limit exceeded")
