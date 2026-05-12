# tests/test_prompts.py
from datetime import datetime, timezone
from pocket_news.models import Article
from pocket_news.prompts import build_synthesis_prompt, LENGTH_TO_WORD_COUNT


def _art(title, outlet, snippet, url="https://example.com"):
    return Article(fetcher="world_news", title=title, url=url, outlet=outlet, snippet=snippet,
                   published_at=datetime(2026, 5, 10, tzinfo=timezone.utc))


def test_length_presets():
    assert LENGTH_TO_WORD_COUNT == {"brief": 300, "standard": 600, "detailed": 1000}


def test_prompt_contains_article_content(three_articles):
    _, user_p = build_synthesis_prompt("EU AI Act", three_articles, "English", 600, "neutral")
    assert "EU AI Act" in user_p
    assert "Reuters" in user_p
    assert "Politico" in user_p
    assert "BBC News" in user_p
    assert "600" in user_p


def test_language_mentioned_multiple_times():
    articles = [_art("Test", "Example", "snippet")]
    _, user_p = build_synthesis_prompt("test", articles, "Spanish", 300, "neutral")
    assert user_p.count("Spanish") >= 3


def test_section_markers_present(three_articles):
    _, user_p = build_synthesis_prompt("test", three_articles, "English", 600, "neutral")
    for marker in ["===HEADLINE===", "===LEAD===", "===BODY===", "===KEY_POINTS==="]:
        assert marker in user_p


def test_articles_block_bounded():
    long_content = "x" * 5000
    articles = [Article(fetcher="world_news", title=f"A{i}", url=f"https://a{i}.com",
                        outlet="Example", content=long_content) for i in range(5)]
    _, user_p = build_synthesis_prompt("test", articles, "English", 600, "neutral")
    assert len(user_p) < 20000
