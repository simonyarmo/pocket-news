# tests/test_synthesizer_unit.py
import httpx
import pytest
from unittest.mock import MagicMock, patch
from pocket_news.synthesizer import OllamaSynthesizer
from pocket_news.exceptions import OllamaUnavailableError

MOCK_TEXT = """===HEADLINE===
EU AI Act Enforcement Underway

===LEAD===
The EU has launched enforcement.

===BODY===
The EU AI Act enforcement began, according to Reuters.

===KEY_POINTS===
- EU AI Act launched
- August 2026 deadline

===SOURCES===
- Reuters
"""


def _mock_response(text):
    msg = MagicMock()
    msg.content = text
    resp = MagicMock()
    resp.message = msg
    return resp


def test_no_think_appended_for_qwen3(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b", disable_thinking_mode=True)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert user_msg["content"].endswith("/no_think")


def test_no_think_not_appended_for_gemma(three_articles):
    synth = OllamaSynthesizer(model="gemma4:latest", disable_thinking_mode=True)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert not user_msg["content"].endswith("/no_think")


def test_no_think_not_appended_when_disabled_false(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b", disable_thinking_mode=False)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert not user_msg["content"].endswith("/no_think")


def test_synthesize_returns_parsed_dict(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b")
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        result = synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        assert result["headline"] == "EU AI Act Enforcement Underway"
        assert len(result["key_points"]) == 2


def test_connect_error_raises_ollama_unavailable(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b")
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.side_effect = httpx.ConnectError("refused")
        with pytest.raises(OllamaUnavailableError, match="ollama serve"):
            synth.synthesize("topic", three_articles, "English", 600, "neutral")
