"""Basic usage — mirrors the README Quick Start."""

from pocket_news import NewsAgent

agent = NewsAgent()
article = agent.research("EU AI Act enforcement")

print(article.headline)
print(article.lead)
print(article.body)
for src in article.sources:
    print(f"- {src.outlet}: {src.url}")
