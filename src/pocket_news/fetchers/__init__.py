"""Fetcher registry — populated as each fetcher module is implemented."""
from .base import BaseFetcher  # noqa: F401
from .mediastack import MediaStackFetcher

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {
    "mediastack": MediaStackFetcher,
}
