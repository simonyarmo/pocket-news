import pytest
from datetime import datetime, timezone


@pytest.fixture
def sample_article():
    from pocket_news.models import Article
    return Article(
        fetcher="world_news",
        title="EU AI Act enforcement begins",
        url="https://reuters.com/technology/eu-ai-act-2026-05-10",
        outlet="Reuters",
        snippet="The EU has started enforcing the AI Act.",
        content="The European Union has begun enforcing the AI Act, marking a significant milestone.",
        published_at=datetime(2026, 5, 10, 14, 32, 0, tzinfo=timezone.utc),
        image_url="https://reuters.com/images/eu-ai.jpg",
    )


@pytest.fixture
def three_articles():
    from pocket_news.models import Article
    return [
        Article(
            fetcher="world_news",
            title="EU AI Act enforcement begins",
            url="https://reuters.com/technology/eu-ai-act-2026-05-10",
            outlet="Reuters",
            snippet="The EU has started enforcing the AI Act.",
            published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
            image_url="https://reuters.com/images/eu-ai.jpg",
        ),
        Article(
            fetcher="the_news",
            title="Companies scramble to comply with EU AI rules",
            url="https://politico.com/news/eu-ai-compliance-2026",
            outlet="Politico",
            snippet="Major tech firms are preparing for EU AI Act compliance.",
            published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
        ),
        Article(
            fetcher="mediastack",
            title="EU regulators publish AI Act guidance",
            url="https://bbc.co.uk/news/technology/eu-ai-guidance",
            outlet="BBC News",
            snippet="European regulators have published detailed guidance.",
            published_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
            image_url="https://bbc.co.uk/images/eu-ai.jpg",
        ),
    ]


@pytest.fixture
def sample_synthesized_article(three_articles):
    from pocket_news.models import SourceCitation, SynthesizedArticle
    return SynthesizedArticle(
        status="ok",
        topic="EU AI Act enforcement",
        output_language="en",
        headline="EU AI Act Enforcement Begins, Companies Race to Comply",
        lead="The EU has launched enforcement of its AI Act. Tech companies scramble to meet the new requirements.",
        body="The EU AI Act enforcement began this week...",
        key_points=[
            "EU AI Act enforcement launched",
            "Companies have until August 2026",
            "High-risk AI systems face strictest rules",
        ],
        featured_image_url="https://reuters.com/images/eu-ai.jpg",
        sources=[
            SourceCitation(
                outlet=a.outlet,
                title=a.title,
                url=a.url,
                published_at=a.published_at,
                fetcher=a.fetcher,
            )
            for a in three_articles
        ],
        model="qwen3:14b",
        generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=3,
        fetcher_status={"world_news": "ok", "the_news": "ok", "mediastack": "ok"},
    )
