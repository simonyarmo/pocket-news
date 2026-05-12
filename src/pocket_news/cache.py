"""Local file cache for SynthesizedArticle results."""
from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Optional

import platformdirs

from .models import SynthesizedArticle

logger = logging.getLogger(__name__)


class CacheStore:
    def __init__(self, cache_dir: Optional[Path], ttl_minutes: int) -> None:
        self._dir = Path(cache_dir) if cache_dir else Path(platformdirs.user_cache_dir("pocket_news"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._ttl = ttl_minutes * 60

    def _key(self, topic: str, language: str, length: str, model: str) -> str:
        return hashlib.sha256(f"{topic.strip().lower()}|{language}|{length}|{model}".encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, topic: str, language: str, length: str, model: str) -> Optional[SynthesizedArticle]:
        path = self._path(self._key(topic, language, length, model))
        try:
            if not path.exists():
                return None
            if time.time() - path.stat().st_mtime > self._ttl:
                return None
            return SynthesizedArticle.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("Cache read failed: %s", exc)
            return None

    def set(self, article: SynthesizedArticle, topic: str, language: str, length: str, model: str) -> None:
        if article.status == "no_results":
            return
        path = self._path(self._key(topic, language, length, model))
        try:
            path.write_text(article.model_dump_json(indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Cache write failed: %s", exc)

    def clear(self) -> int:
        count = 0
        for f in self._dir.glob("*.json"):
            f.unlink(missing_ok=True)
            count += 1
        return count
