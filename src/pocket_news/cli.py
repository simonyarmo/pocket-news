"""Command-line interface for pocket-news."""
from __future__ import annotations
import argparse
import logging
import pathlib
import sys


def _run_setup() -> None:
    secrets_path = pathlib.Path(__file__).parent / "_secrets.py"
    print("pocket-news API Key Setup")
    print("=" * 40)
    print("Sign up for free at:")
    print("  World News API : https://worldnewsapi.com  (50 pts/day)")
    print("  The News API   : https://thenewsapi.com    (100 req/day)")
    print("  MediaStack     : https://mediastack.com    (100 req/month)")
    print()
    print("Press Enter to skip a key (that fetcher will be disabled).")
    print()
    world = input("World News API key: ").strip()
    the_news = input("The News API key:   ").strip()
    mediastack = input("MediaStack key:    ").strip()
    content = (
        '"""pocket-news API keys — NOT checked into git."""\n'
        f'WORLD_NEWS_API_KEY = "{world}"\n'
        f'THE_NEWS_API_KEY = "{the_news}"\n'
        f'MEDIASTACK_API_KEY = "{mediastack}"\n'
    )
    secrets_path.write_text(content, encoding="utf-8")
    print(f"\nKeys saved to {secrets_path}")
    print('Run: python -m pocket_news "your topic here"')


def _format_article(article) -> str:
    lines = [
        f"\n{'=' * 70}",
        article.headline,
        f"{'=' * 70}",
        "",
        article.lead,
        "",
        article.body,
        "",
        "KEY POINTS:",
    ]
    for pt in article.key_points:
        lines.append(f"  • {pt}")
    lines += ["", "SOURCES:"]
    for src in article.sources:
        lines.append(f"  [{src.fetcher}] {src.outlet}: {src.url}")
    lines.append(
        f"\nStatus: {article.status} | Model: {article.model} | Articles: {article.article_count}"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pocket-news",
        description="Fetch and synthesize news with a local LLM.",
    )
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--language", default="en", help="Output language name or ISO code")
    parser.add_argument(
        "--length", choices=["brief", "standard", "detailed"], default="standard"
    )
    parser.add_argument("--model", help="Ollama model override (e.g. gemma4:latest)")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output raw JSON")
    parser.add_argument("--no-cache", action="store_true", help="Disable cache for this run")
    parser.add_argument("--setup", action="store_true", help="Interactive API key setup")
    args = parser.parse_args()

    if args.setup:
        _run_setup()
        return

    if not args.topic:
        parser.print_help()
        sys.exit(1)

    logging.basicConfig(level=logging.WARNING)

    from .agent import NewsAgent
    from .config import AgentConfig

    kwargs: dict = {}
    if args.model:
        kwargs["ollama_model"] = args.model
    if args.no_cache:
        kwargs["cache_enabled"] = False

    config = AgentConfig(**kwargs) if kwargs else AgentConfig()
    agent = NewsAgent(config=config)
    article = agent.research(args.topic, language=args.language, length=args.length)

    if args.as_json:
        print(article.model_dump_json(indent=2))
    else:
        print(_format_article(article))
