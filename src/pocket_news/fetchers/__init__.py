"""Fetcher registry — populated as each fetcher module is implemented."""
from .base import BaseFetcher  # noqa: F401
from .mediastack import MediaStackFetcher
from .the_news import TheNewsAPIFetcher
from .world_news import WorldNewsFetcher

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {
    "mediastack": MediaStackFetcher,
    "the_news": TheNewsAPIFetcher,
    "world_news": WorldNewsFetcher,
}
