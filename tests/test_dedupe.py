# tests/test_dedupe.py
from pocket_news.dedupe import ArticleDeduplicator
from pocket_news.models import Article


def _art(fetcher="world_news", title="Title", url="https://example.com", content=None, snippet=None):
    return Article(fetcher=fetcher, title=title, url=url, outlet="Example", content=content, snippet=snippet)


def test_empty_list():
    assert ArticleDeduplicator.deduplicate([]) == []


def test_no_duplicates_unchanged():
    articles = [_art(url="https://a.com", title="Article A"), _art(url="https://b.com", title="Article B")]
    assert len(ArticleDeduplicator.deduplicate(articles)) == 2


def test_exact_url_dedup():
    a1 = _art(url="https://example.com/article?utm_source=tw", title="T1")
    a2 = _art(url="https://example.com/article?ref=fb", title="T1")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_url_trailing_slash_dedup():
    a1 = _art(url="https://example.com/article/")
    a2 = _art(url="https://example.com/article")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_exact_title_dedup():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins")
    a2 = _art(url="https://b.com", title="EU AI Act enforcement begins")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_similar_title_dedup():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins across Europe")
    a2 = _art(url="https://b.com", title="EU AI Act enforcement begins in Europe")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_dissimilar_titles_kept():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins")
    a2 = _art(url="https://b.com", title="SpaceX launches Starship")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 2


def test_prefers_content_over_snippet():
    with_content = _art(url="https://a.com", title="Same Title", content="Full article body.")
    snippet_only = _art(url="https://b.com", title="Same Title", snippet="Short snippet.")
    result = ArticleDeduplicator.deduplicate([snippet_only, with_content])
    assert len(result) == 1
    assert result[0].content == "Full article body."


def test_prefers_longer_content():
    short = _art(url="https://a.com", title="Same", content="Short.")
    long = _art(url="https://b.com", title="Same", content="A much longer article body with more details.")
    result = ArticleDeduplicator.deduplicate([short, long])
    assert len(result) == 1
    assert "longer" in result[0].content
