"""Article deduplication — URL canonicalization + title similarity."""
from __future__ import annotations
import difflib
from urllib.parse import urlparse, urlunparse
from .models import Article

_SIMILARITY_THRESHOLD = 0.85


def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        norm = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower(), query="", fragment="")
        return urlunparse(norm._replace(path=norm.path.rstrip("/")))
    except Exception:
        return url.lower()


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _prefer(a: Article, b: Article) -> Article:
    a_has = bool(a.content)
    b_has = bool(b.content)
    if a_has != b_has:
        return a if a_has else b
    return a if len(a.content or a.snippet or "") >= len(b.content or b.snippet or "") else b


class ArticleDeduplicator:
    """Deduplicates a list of Articles by URL canonicalization and title similarity."""

    @staticmethod
    def deduplicate(articles: list[Article]) -> list[Article]:
        kept: list[Article] = []
        canonical_urls: set[str] = set()
        norm_titles: list[str] = []

        for article in articles:
            canon = _canonical_url(article.url)
            norm = _normalize_title(article.title)

            if canon in canonical_urls:
                idx = next((i for i, a in enumerate(kept) if _canonical_url(a.url) == canon), None)
                if idx is not None:
                    kept[idx] = _prefer(kept[idx], article)
                continue

            matched = None
            for i, t in enumerate(norm_titles):
                if t == norm or difflib.SequenceMatcher(None, norm, t).ratio() >= _SIMILARITY_THRESHOLD:
                    matched = i
                    break

            if matched is not None:
                kept[matched] = _prefer(kept[matched], article)
                continue

            canonical_urls.add(canon)
            norm_titles.append(norm)
            kept.append(article)

        return kept
