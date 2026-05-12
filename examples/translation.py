"""Synthesize the same topic in two languages."""

from pocket_news import NewsAgent

agent = NewsAgent()
topic = "EU AI Act enforcement"

for lang in ["Spanish", "French"]:
    article = agent.research(topic, language=lang, length="brief")
    print(f"\n=== {lang} ({article.output_language}) ===")
    print(article.headline)
    print(article.lead)
