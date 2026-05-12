"""Fetcher registry — populated as each fetcher module is implemented."""
from .base import BaseFetcher  # noqa: F401

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {}
