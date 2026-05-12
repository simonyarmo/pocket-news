"""Parent-project pattern: batch research over user interests."""

from pocket_news import NewsAgent

agent = NewsAgent()
interests = ["EU AI Act enforcement", "SpaceX Starship", "Federal Reserve rate decision"]
articles = agent.research_batch(interests, max_workers=3)

for article in articles:
    if article.status == "no_results":
        print(f"[no results] {article.topic}")
        continue
    print(f"\n{'=' * 60}\n[{article.status}] {article.headline}")
    print(f"Sources: {article.article_count} | Language: {article.output_language}")
    print(article.lead)
