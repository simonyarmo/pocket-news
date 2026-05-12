# tests/test_base_fetcher.py
from pocket_news.fetchers.base import resolve_outlet, _strip_html, DOMAIN_TO_OUTLET


def test_known_domain():
    name, domain = resolve_outlet("https://www.reuters.com/article/123")
    assert name == "Reuters"
    assert domain == "reuters.com"


def test_bbc_co_uk():
    name, domain = resolve_outlet("https://www.bbc.co.uk/news/tech")
    assert name == "BBC News"
    assert domain == "bbc.co.uk"


def test_unknown_domain_titlecased():
    name, domain = resolve_outlet("https://somesite.example.com/article")
    # Falls back: first non-www segment titlecased
    assert isinstance(name, str)
    assert len(name) > 0


def test_plain_domain_string():
    name, domain = resolve_outlet("reuters.com")
    assert name == "Reuters"


def test_strip_html_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_entities():
    result = _strip_html("AT&amp;T")
    assert "&amp;" not in result
    assert "AT" in result


def test_strip_html_plain_passthrough():
    assert _strip_html("No HTML here") == "No HTML here"
