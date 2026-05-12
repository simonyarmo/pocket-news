"""Prompt templates and builder for OllamaSynthesizer."""
from __future__ import annotations
from .models import Article

LENGTH_TO_WORD_COUNT: dict[str, int] = {"brief": 300, "standard": 600, "detailed": 1000}

SYSTEM_PROMPT = """You are a professional news editor. Given a set of articles from different outlets covering the same topic, your job is to synthesize them into a single coherent news article that reads as one cohesive piece — not a list of summaries.

Guidelines:
- Use ONLY information from the provided articles. Do not add facts from your own knowledge.
- If outlets disagree on a fact, note the disagreement explicitly.
- Attribute facts inline: "according to <outlet>" or "(<outlet>)".
- Write in a neutral, wire-service tone. No editorializing.
- If articles lack enough information, write shorter rather than padding.
- Output MUST follow the section-delimited format exactly. No commentary before or after.
- Do not include the section markers inside any section's content.
"""

SYNTHESIS_USER_TEMPLATE = """TOPIC: {topic}

OUTPUT LANGUAGE: {output_language_name}
TONE: {tone}
TARGET BODY LENGTH: approximately {target_word_count} words

ARTICLES ({n_articles} total):
{articles_block}

Write a synthesized news article entirely in {output_language_name}, following the exact output format below. The section markers (===HEADLINE===, ===LEAD===, etc.) must remain in English/ASCII. Everything else should be in {output_language_name}.

===HEADLINE===
<headline in {output_language_name}>

===LEAD===
<2-3 sentence lead paragraph in {output_language_name}>

===BODY===
<full article body in {output_language_name}, ~{target_word_count} words>

===KEY_POINTS===
- <point in {output_language_name}>
- <point in {output_language_name}>
- <point in {output_language_name}>

===SOURCES===
- <outlet name>
- <outlet name>
"""

_MAX_ARTICLES_CHARS = 12_000
_MAX_CONTENT_CHARS = 4_000


def _format_article(n: int, article: Article) -> str:
    date_str = article.published_at.isoformat() if article.published_at else "unknown date"
    body = article.content[:_MAX_CONTENT_CHARS] if article.content else (article.snippet or "")
    return f"[{n}] {article.outlet} — {date_str} — {article.url}\nTITLE: {article.title}\n{body}\n---"


def build_synthesis_prompt(
    topic: str,
    articles: list[Article],
    output_language_name: str,
    target_word_count: int,
    tone: str,
) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt)."""
    blocks: list[str] = []
    total = 0
    for i, article in enumerate(articles, 1):
        block = _format_article(i, article)
        if total + len(block) > _MAX_ARTICLES_CHARS:
            break
        blocks.append(block)
        total += len(block)

    user_prompt = SYNTHESIS_USER_TEMPLATE.format(
        topic=topic, output_language_name=output_language_name, tone=tone,
        target_word_count=target_word_count, n_articles=len(blocks),
        articles_block="\n\n".join(blocks),
    )
    return SYSTEM_PROMPT, user_prompt
