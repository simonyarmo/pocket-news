# pocket-news Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `pocket-news`, a pip-installable Python library (`from pocket_news import NewsAgent`) that fetches news from three APIs in parallel, deduplicates, and synthesizes into one article via a local Ollama LLM with optional translation and base64 image embedding.

**Architecture:** A thin `NewsAgent` orchestrator coordinates parallel fetching (`ThreadPoolExecutor`), deduplication (`ArticleDeduplicator`), synthesis (`OllamaSynthesizer`), and local file caching (`CacheStore`). All data flows as Pydantic v2 models. Public API is synchronous.

**Tech Stack:** Python 3.9+, `httpx`, `pydantic>=2`, `ollama>=0.4`, `python-dateutil`, `platformdirs`, `respx` (test HTTP mocking), `pytest`

---

## File Map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | Package metadata, deps, tool config |
| `src/pocket_news/__init__.py` | Public exports, `__version__`, `configure_logging()` |
| `src/pocket_news/exceptions.py` | `PocketNewsError` hierarchy |
| `src/pocket_news/languages.py` | ISO 639-1 normalization |
| `src/pocket_news/models.py` | `Article`, `SourceCitation`, `SynthesizedArticle` |
| `src/pocket_news/config.py` | `AgentConfig` with key resolution chain |
| `src/pocket_news/fetchers/base.py` | `BaseFetcher`, `DOMAIN_TO_OUTLET`, `resolve_outlet()` |
| `src/pocket_news/fetchers/__init__.py` | `FETCHER_REGISTRY` |
| `src/pocket_news/fetchers/mediastack.py` | `MediaStackFetcher` |
| `src/pocket_news/fetchers/the_news.py` | `TheNewsAPIFetcher` |
| `src/pocket_news/fetchers/world_news.py` | `WorldNewsFetcher` |
| `src/pocket_news/dedupe.py` | `ArticleDeduplicator` |
| `src/pocket_news/images.py` | `fetch_and_encode_image()` |
| `src/pocket_news/cache.py` | `CacheStore` |
| `src/pocket_news/prompts.py` | Prompt templates, `build_synthesis_prompt()` |
| `src/pocket_news/synthesizer.py` | `OllamaSynthesizer` |
| `src/pocket_news/agent.py` | `NewsAgent` |
| `src/pocket_news/cli.py` | `argparse` CLI + `--setup` command |
| `src/pocket_news/__main__.py` | Entry point → `cli.main()` |
| `src/pocket_news/_secrets.example.py` | API key template (checked in) |
| `tests/conftest.py` | Shared pytest fixtures |
| `tests/fixtures/mediastack_response.json` | Saved MediaStack response |
| `tests/fixtures/the_news_response.json` | Saved The News API response |
| `tests/fixtures/world_news_response.json` | Saved World News API response |
| `tests/test_exceptions.py` | Exception hierarchy smoke test |
| `tests/test_languages.py` | `normalize_language()` |
| `tests/test_models.py` | Model validation + JSON round-trip |
| `tests/test_config.py` | Key resolution, validators, `resolved_fetchers()` |
| `tests/test_base_fetcher.py` | `resolve_outlet()`, HTML stripping |
| `tests/test_fetcher_mediastack.py` | MediaStack unit + integration |
| `tests/test_fetcher_the_news.py` | The News API unit + integration |
| `tests/test_fetcher_world_news.py` | World News API unit + integration |
| `tests/test_dedupe.py` | Dedup edge cases |
| `tests/test_images.py` | Image fetch, encode, error paths |
| `tests/test_cache.py` | Cache round-trip, TTL, no-results skip |
| `tests/test_prompts.py` | Prompt building, language injection |
| `tests/test_parser.py` | Section parsing, `<think>` stripping |
| `tests/test_synthesizer_unit.py` | `OllamaSynthesizer` with mocked client |
| `tests/test_agent.py` | Full pipeline with mocked components |
| `examples/basic_usage.py` | Quick start |
| `examples/translation.py` | Spanish + French output |
| `examples/multi_interest.py` | Batch research pattern |

---

## Task 1: Project scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `src/pocket_news/__init__.py` (stub)
- Create: `src/pocket_news/exceptions.py` (stub)
- Create: `src/pocket_news/languages.py` (stub)
- Create: `src/pocket_news/models.py` (stub)
- Create: `src/pocket_news/config.py` (stub)
- Create: `src/pocket_news/agent.py` (stub)
- Create: `src/pocket_news/cache.py` (stub)
- Create: `src/pocket_news/dedupe.py` (stub)
- Create: `src/pocket_news/images.py` (stub)
- Create: `src/pocket_news/prompts.py` (stub)
- Create: `src/pocket_news/synthesizer.py` (stub)
- Create: `src/pocket_news/cli.py` (stub)
- Create: `src/pocket_news/__main__.py`
- Create: `src/pocket_news/_secrets.example.py`
- Create: `src/pocket_news/fetchers/__init__.py` (stub)
- Create: `src/pocket_news/fetchers/base.py` (stub)
- Create: `src/pocket_news/fetchers/mediastack.py` (stub)
- Create: `src/pocket_news/fetchers/the_news.py` (stub)
- Create: `src/pocket_news/fetchers/world_news.py` (stub)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/fixtures/` (empty dir, add `.gitkeep`)
- Create: `examples/` (empty dir)

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "pocket-news"
version = "0.1.0"
description = "Fetch news from multiple sources and synthesize with a local LLM"
readme = "README.md"
requires-python = ">=3.9"
license = {text = "MIT"}
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.0",
    "ollama>=0.4",
    "python-dateutil>=2.9",
    "platformdirs>=4.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "respx>=0.21",
    "ruff>=0.4",
    "mypy>=1.10",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.ruff]
line-length = 100

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.9"
```

- [ ] **Step 2: Create `.gitignore`**

```
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg
src/pocket_news/_secrets.py
```

- [ ] **Step 3: Create `LICENSE`**

```
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 4: Create `src/pocket_news/_secrets.example.py`**

```python
"""Copy this file to _secrets.py and fill in your keys, or run: python -m pocket_news --setup"""
WORLD_NEWS_API_KEY = ""
THE_NEWS_API_KEY = ""
MEDIASTACK_API_KEY = ""
```

- [ ] **Step 5: Create `src/pocket_news/__main__.py`**

```python
from .cli import main

if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Create stub files** — each file gets a single docstring, nothing else:

`src/pocket_news/__init__.py`:
```python
"""pocket-news — fetch, deduplicate, and synthesize news with a local LLM."""
__version__ = "0.1.0"
```

`src/pocket_news/exceptions.py`, `languages.py`, `models.py`, `config.py`, `agent.py`, `cache.py`, `dedupe.py`, `images.py`, `prompts.py`, `synthesizer.py`, `cli.py` — each gets just a module docstring.

`src/pocket_news/fetchers/__init__.py`, `fetchers/base.py`, `fetchers/mediastack.py`, `fetchers/the_news.py`, `fetchers/world_news.py` — each gets just a module docstring.

- [ ] **Step 7: Create `tests/__init__.py`** (empty) and `tests/fixtures/.gitkeep` (empty).

- [ ] **Step 8: Create `tests/conftest.py`**

```python
import pytest
from datetime import datetime, timezone
from pocket_news.models import Article, SourceCitation, SynthesizedArticle


@pytest.fixture
def sample_article() -> Article:
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
def three_articles() -> list:
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
def sample_synthesized_article(three_articles) -> SynthesizedArticle:
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
```

- [ ] **Step 9: Install and verify**

```bash
pip install -e ".[dev]"
python -c "import pocket_news; print(pocket_news.__version__)"
```

Expected: `0.1.0`

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml .gitignore LICENSE src/ tests/ examples/
git commit -m "feat: project scaffold — pocket-news 0.1.0"
```

---

## Task 2: Exceptions

**Files:**
- Create: `src/pocket_news/exceptions.py`
- Create: `tests/test_exceptions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_exceptions.py
import pytest
from pocket_news.exceptions import (
    PocketNewsError,
    ConfigurationError,
    FetcherError,
    RateLimitError,
    OllamaUnavailableError,
    SynthesisParseError,
)


def test_hierarchy():
    assert issubclass(ConfigurationError, PocketNewsError)
    assert issubclass(FetcherError, PocketNewsError)
    assert issubclass(RateLimitError, FetcherError)
    assert issubclass(OllamaUnavailableError, PocketNewsError)
    assert issubclass(SynthesisParseError, PocketNewsError)


def test_catch_as_base():
    with pytest.raises(PocketNewsError):
        raise OllamaUnavailableError("Ollama not running")

    with pytest.raises(PocketNewsError):
        raise RateLimitError("Rate limit exceeded")
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_exceptions.py -v
```

Expected: `ImportError` — names not defined yet.

- [ ] **Step 3: Implement `src/pocket_news/exceptions.py`**

```python
"""Exception hierarchy for pocket-news."""


class PocketNewsError(Exception):
    """Base exception for pocket-news."""


class ConfigurationError(PocketNewsError):
    """Invalid configuration value."""


class FetcherError(PocketNewsError):
    """A single fetcher failed."""


class RateLimitError(FetcherError):
    """API rate limit exceeded."""


class OllamaUnavailableError(PocketNewsError):
    """Cannot reach Ollama or the configured model is not pulled."""


class SynthesisParseError(PocketNewsError):
    """Model returned output that could not be parsed into required sections."""
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_exceptions.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/exceptions.py tests/test_exceptions.py
git commit -m "feat: exception hierarchy"
```

---

## Task 3: Language normalization

**Files:**
- Create: `src/pocket_news/languages.py`
- Create: `tests/test_languages.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_languages.py
import pytest
from pocket_news.languages import normalize_language


def test_name_mixed_case():
    iso, display = normalize_language("Spanish")
    assert iso == "es"
    assert display == "Spanish"


def test_name_lowercase():
    iso, display = normalize_language("spanish")
    assert iso == "es"


def test_name_uppercase():
    iso, display = normalize_language("SPANISH")
    assert iso == "es"


def test_iso_code():
    iso, display = normalize_language("es")
    assert iso == "es"
    assert display == "Spanish"


def test_mandarin_alias():
    iso, display = normalize_language("Mandarin")
    assert iso == "zh"


def test_english_default():
    iso, display = normalize_language("en")
    assert iso == "en"
    assert display == "English"


def test_unknown_raises():
    with pytest.raises(ValueError, match="Unrecognized language"):
        normalize_language("Klingon")


def test_whitespace_stripped():
    iso, display = normalize_language("  French  ")
    assert iso == "fr"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_languages.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/languages.py`**

```python
"""ISO 639-1 language normalization."""

LANGUAGE_NAME_TO_ISO: dict[str, str] = {
    "english": "en", "spanish": "es", "french": "fr", "german": "de",
    "italian": "it", "portuguese": "pt", "dutch": "nl", "russian": "ru",
    "chinese": "zh", "mandarin": "zh", "japanese": "ja", "korean": "ko",
    "arabic": "ar", "hindi": "hi", "turkish": "tr", "polish": "pl",
    "vietnamese": "vi", "thai": "th", "indonesian": "id", "swedish": "sv",
    "norwegian": "no", "danish": "da", "finnish": "fi", "greek": "el",
    "hebrew": "he", "czech": "cs", "hungarian": "hu", "romanian": "ro",
    "ukrainian": "uk",
}

ISO_TO_DISPLAY_NAME: dict[str, str] = {
    v: k.capitalize() for k, v in LANGUAGE_NAME_TO_ISO.items()
}


def normalize_language(lang: str) -> tuple[str, str]:
    """
    Returns (iso_code, display_name).

    Accepts language names (case-insensitive) or 2-letter ISO 639-1 codes.
    Raises ValueError for unrecognized inputs.
    """
    lang = lang.strip()
    lower = lang.lower()

    if lower in LANGUAGE_NAME_TO_ISO:
        iso = LANGUAGE_NAME_TO_ISO[lower]
        return iso, ISO_TO_DISPLAY_NAME[iso]

    if lower in ISO_TO_DISPLAY_NAME:
        return lower, ISO_TO_DISPLAY_NAME[lower]

    raise ValueError(
        f"Unrecognized language: {lang!r}. "
        "Use a name (e.g. 'Spanish') or ISO 639-1 code (e.g. 'es')."
    )
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_languages.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/languages.py tests/test_languages.py
git commit -m "feat: language normalization"
```

---

## Task 4: Data models

**Files:**
- Create: `src/pocket_news/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
import pytest
from datetime import datetime, timezone
from pocket_news.models import Article, SourceCitation, SynthesizedArticle


def test_article_minimal():
    a = Article(fetcher="world_news", title="Test", url="https://example.com", outlet="Example")
    assert a.content is None
    assert a.categories == []


def test_article_json_roundtrip():
    a = Article(
        fetcher="the_news",
        title="Test Article",
        url="https://example.com/article",
        outlet="Example News",
        published_at=datetime(2026, 5, 10, tzinfo=timezone.utc),
    )
    json_str = a.model_dump_json()
    restored = Article.model_validate_json(json_str)
    assert restored.title == a.title
    assert restored.published_at == a.published_at


def test_synthesized_article_defaults():
    from pocket_news.models import SourceCitation
    art = SynthesizedArticle(
        topic="Test",
        output_language="en",
        headline="Headline",
        lead="Lead paragraph.",
        body="Body text.",
        key_points=["Point 1"],
        model="qwen3:14b",
        generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=1,
        fetcher_status={"world_news": "ok"},
    )
    assert art.status == "ok"
    assert art.featured_image_b64 is None
    assert art.sources == []


def test_synthesized_article_json_roundtrip(sample_synthesized_article):
    json_str = sample_synthesized_article.model_dump_json()
    restored = SynthesizedArticle.model_validate_json(json_str)
    assert restored.headline == sample_synthesized_article.headline
    assert len(restored.sources) == 3
    assert restored.status == "ok"


def test_no_results_status():
    art = SynthesizedArticle(
        status="no_results",
        topic="obscure topic",
        output_language="en",
        headline="",
        lead="",
        body="",
        key_points=[],
        model="qwen3:14b",
        generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=0,
        fetcher_status={"world_news": "empty", "the_news": "empty", "mediastack": "empty"},
    )
    assert art.status == "no_results"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/models.py`**

```python
"""Core Pydantic v2 data models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel


class Article(BaseModel):
    """A single news article returned by a fetcher."""

    fetcher: str
    source_id: str | None = None
    title: str
    url: str
    snippet: str | None = None
    content: str | None = None
    author: str | None = None
    outlet: str
    outlet_domain: str | None = None
    language: str | None = None
    country: str | None = None
    categories: list[str] = []
    published_at: datetime | None = None
    image_url: str | None = None


class SourceCitation(BaseModel):
    """Attribution record for one input article."""

    outlet: str
    title: str
    url: str
    published_at: datetime | None = None
    fetcher: str


class SynthesizedArticle(BaseModel):
    """Final output of NewsAgent.research()."""

    status: Literal["ok", "no_results", "partial"] = "ok"
    topic: str
    output_language: str

    headline: str
    lead: str
    body: str
    key_points: list[str]

    featured_image_url: str | None = None
    featured_image_b64: str | None = None
    featured_image_mime: str | None = None

    sources: list[SourceCitation] = []
    model: str
    generated_at: datetime
    article_count: int
    fetcher_status: dict[str, str]
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_models.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/models.py tests/test_models.py
git commit -m "feat: Article, SourceCitation, SynthesizedArticle models"
```

---

## Task 5: AgentConfig

**Files:**
- Create: `src/pocket_news/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import os
import pytest
from pocket_news.config import AgentConfig
from pocket_news.exceptions import ConfigurationError


def test_defaults():
    config = AgentConfig()
    assert config.ollama_model == "qwen3:14b"
    assert config.max_articles_per_source == 5
    assert config.tone == "neutral"
    assert config.cache_enabled is True


def test_env_var_key_resolution(monkeypatch):
    monkeypatch.setenv("WORLD_NEWS_API_KEY", "env-key-123")
    config = AgentConfig()
    assert config.world_news_api_key == "env-key-123"


def test_kwarg_overrides_env(monkeypatch):
    monkeypatch.setenv("WORLD_NEWS_API_KEY", "env-key")
    config = AgentConfig(world_news_api_key="kwarg-key")
    assert config.world_news_api_key == "kwarg-key"


def test_missing_key_is_none(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig()
    # All None is fine — fetchers are disabled silently
    assert config.world_news_api_key is None


def test_max_articles_out_of_range():
    with pytest.raises(ConfigurationError):
        AgentConfig(max_articles_per_source=100)


def test_lookback_days_out_of_range():
    with pytest.raises(ConfigurationError):
        AgentConfig(lookback_days=0)


def test_invalid_tone():
    with pytest.raises(ConfigurationError):
        AgentConfig(tone="sarcastic")


def test_negative_cache_ttl():
    with pytest.raises(ConfigurationError):
        AgentConfig(cache_ttl_minutes=-1)


def test_resolved_fetchers_filters_none(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig(world_news_api_key="key-a", the_news_api_key=None, mediastack_api_key=None)
    result = config.resolved_fetchers()
    assert len(result) == 1
    assert result[0][0] == "world_news"


def test_resolved_fetchers_respects_enabled_list(monkeypatch):
    monkeypatch.delenv("WORLD_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("THE_NEWS_API_KEY", raising=False)
    monkeypatch.delenv("MEDIASTACK_API_KEY", raising=False)
    config = AgentConfig(
        world_news_api_key="key-a",
        the_news_api_key="key-b",
        mediastack_api_key="key-c",
        enabled_fetchers=["world_news"],
    )
    result = config.resolved_fetchers()
    assert [fid for fid, _ in result] == ["world_news"]
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/config.py`**

```python
"""AgentConfig — configuration with API key resolution chain."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .exceptions import ConfigurationError


def _resolve_key(name: str) -> str | None:
    val = os.getenv(name)
    if val:
        return val
    try:
        from . import _secrets  # type: ignore[import]

        val = getattr(_secrets, name, None)
        if val:
            return val
    except ImportError:
        pass
    return None


class AgentConfig(BaseModel):
    """All configuration knobs for NewsAgent. Every field has a sensible default."""

    world_news_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("WORLD_NEWS_API_KEY")
    )
    the_news_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("THE_NEWS_API_KEY")
    )
    mediastack_api_key: str | None = Field(
        default_factory=lambda: _resolve_key("MEDIASTACK_API_KEY")
    )

    enabled_fetchers: list[str] | None = None
    max_articles_per_source: int = 5
    lookback_days: int = 7
    fetch_timeout_seconds: float = 15.0

    fetch_image_data: bool = True
    image_max_bytes: int = 2_000_000
    image_fetch_timeout: float = 5.0

    cache_enabled: bool = True
    cache_ttl_minutes: int = 60
    cache_dir: Any = None  # Path | None

    ollama_model: str = "qwen3:14b"
    ollama_host: str = "http://localhost:11434"
    ollama_timeout_seconds: int = 120
    disable_thinking_mode: bool = True

    tone: str = "neutral"

    @model_validator(mode="after")
    def _validate_ranges(self) -> "AgentConfig":
        if not 1 <= self.max_articles_per_source <= 50:
            raise ConfigurationError("max_articles_per_source must be between 1 and 50")
        if not 1 <= self.lookback_days <= 30:
            raise ConfigurationError("lookback_days must be between 1 and 30")
        if self.tone not in {"neutral", "analytical", "brief"}:
            raise ConfigurationError("tone must be 'neutral', 'analytical', or 'brief'")
        if self.cache_ttl_minutes < 0:
            raise ConfigurationError("cache_ttl_minutes must be >= 0")
        if self.image_max_bytes <= 0:
            raise ConfigurationError("image_max_bytes must be > 0")
        return self

    def resolved_fetchers(self) -> list[tuple[str, str]]:
        """Returns (fetcher_id, api_key) pairs for fetchers that are enabled and have keys."""
        candidates = [
            ("world_news", self.world_news_api_key),
            ("the_news", self.the_news_api_key),
            ("mediastack", self.mediastack_api_key),
        ]
        return [
            (fid, key)
            for fid, key in candidates
            if key and (self.enabled_fetchers is None or fid in self.enabled_fetchers)
        ]
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/config.py tests/test_config.py
git commit -m "feat: AgentConfig with key resolution and range validators"
```

---

## Task 6: Fetcher base and registry

**Files:**
- Create: `src/pocket_news/fetchers/base.py`
- Modify: `src/pocket_news/fetchers/__init__.py`
- Create: `tests/test_base_fetcher.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_base_fetcher.py
from pocket_news.fetchers.base import resolve_outlet, _strip_html, DOMAIN_TO_OUTLET


def test_known_domain():
    name, domain = resolve_outlet("https://www.reuters.com/article/123")
    assert name == "Reuters"
    assert domain == "reuters.com"


def test_bbc_co_uk():
    name, domain = resolve_outlet("https://www.bbc.co.uk/news/tech")
    assert name == "BBC News"
    assert domain == "bbc.co.uk"


def test_unknown_domain_titlecased():
    name, domain = resolve_outlet("https://somesite.example.com/article")
    assert name == "Somesite"  # first non-www segment, titlecased


def test_plain_domain_string():
    name, domain = resolve_outlet("reuters.com")
    assert name == "Reuters"


def test_strip_html_tags():
    assert _strip_html("<p>Hello <b>world</b></p>") == "Hello world"


def test_strip_html_entities():
    assert _strip_html("AT&amp;T &mdash; news") == "AT T   news"


def test_strip_html_plain_passthrough():
    assert _strip_html("No HTML here") == "No HTML here"
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_base_fetcher.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/fetchers/base.py`**

```python
"""BaseFetcher abstract class and shared outlet utilities."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.parse import urlparse

import httpx

from ..models import Article

DOMAIN_TO_OUTLET: dict[str, str] = {
    "bbc.co.uk": "BBC News",
    "bbc.com": "BBC News",
    "nytimes.com": "The New York Times",
    "washingtonpost.com": "The Washington Post",
    "theguardian.com": "The Guardian",
    "reuters.com": "Reuters",
    "apnews.com": "Associated Press",
    "npr.org": "NPR",
    "cnn.com": "CNN",
    "foxnews.com": "Fox News",
    "wsj.com": "The Wall Street Journal",
    "bloomberg.com": "Bloomberg",
    "ft.com": "Financial Times",
    "aljazeera.com": "Al Jazeera",
    "dw.com": "Deutsche Welle",
    "france24.com": "France 24",
    "economist.com": "The Economist",
    "politico.com": "Politico",
    "axios.com": "Axios",
    "theverge.com": "The Verge",
    "techcrunch.com": "TechCrunch",
    "arstechnica.com": "Ars Technica",
    "nypost.com": "New York Post",
    "nbcnews.com": "NBC News",
    "cbsnews.com": "CBS News",
    "abcnews.go.com": "ABC News",
}

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&[a-zA-Z]+;|&#\d+;|&#x[0-9a-fA-F]+;")


def _strip_html(text: str) -> str:
    text = _HTML_TAG.sub("", text)
    text = _HTML_ENTITY.sub(" ", text)
    return text.strip()


def resolve_outlet(url_or_domain: str) -> tuple[str, str]:
    """Returns (outlet_name, domain). Falls back to titlecased hostname segment."""
    raw = url_or_domain if "://" in url_or_domain else "https://" + url_or_domain
    try:
        parsed = urlparse(raw)
        domain = parsed.netloc.lower()
    except Exception:
        domain = url_or_domain.lower()

    if domain.startswith("www."):
        domain = domain[4:]

    if domain in DOMAIN_TO_OUTLET:
        return DOMAIN_TO_OUTLET[domain], domain

    parts = domain.split(".")
    for i in range(1, len(parts)):
        candidate = ".".join(parts[i:])
        if candidate in DOMAIN_TO_OUTLET:
            return DOMAIN_TO_OUTLET[candidate], domain

    base = parts[0].replace("-", " ").title() if parts else domain.title()
    return base, domain


class BaseFetcher(ABC):
    """Abstract base for all news API fetchers."""

    id: ClassVar[str]
    display_name: ClassVar[str]

    def __init__(self, api_key: str, timeout: float = 15.0) -> None:
        self.api_key = api_key
        self._client = httpx.Client(timeout=timeout)

    @abstractmethod
    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        """Fetch articles for topic. Raises FetcherError or RateLimitError on failure."""
        ...

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Update `src/pocket_news/fetchers/__init__.py`** (stub until fetcher tasks add entries):

```python
"""Fetcher registry — populated as each fetcher module is implemented."""
from .base import BaseFetcher  # noqa: F401

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {}
```

- [ ] **Step 5: Run to verify tests pass**

```bash
pytest tests/test_base_fetcher.py -v
```

Expected: 7 passed.

- [ ] **Step 6: Commit**

```bash
git add src/pocket_news/fetchers/ tests/test_base_fetcher.py
git commit -m "feat: BaseFetcher, DOMAIN_TO_OUTLET, resolve_outlet, HTML stripping"
```

---

## Task 7: MediaStack fetcher

**Files:**
- Create: `tests/fixtures/mediastack_response.json`
- Create: `tests/test_fetcher_mediastack.py`
- Modify: `src/pocket_news/fetchers/mediastack.py`
- Modify: `src/pocket_news/fetchers/__init__.py`

- [ ] **Step 1: Create `tests/fixtures/mediastack_response.json`**

```json
{
  "pagination": {"limit": 2, "offset": 0, "count": 2, "total": 15},
  "data": [
    {
      "author": "John Smith",
      "title": "EU AI Act enforcement: first cases expected",
      "description": "European regulators are expected to begin enforcement actions under the EU AI Act.",
      "url": "https://bbc.co.uk/news/technology/eu-ai-enforcement-2026",
      "source": "BBC News",
      "image": "https://bbc.co.uk/images/eu-ai-enforcement.jpg",
      "category": "technology",
      "language": "en",
      "country": "gb",
      "published_at": "2026-05-10T15:30:00+00:00"
    },
    {
      "author": null,
      "title": "AI companies prepare for EU compliance deadline",
      "description": "Major AI firms including Google and Microsoft are preparing for EU compliance.",
      "url": "https://theguardian.com/technology/eu-ai-compliance-deadline",
      "source": "The Guardian",
      "image": null,
      "category": "technology",
      "language": "en",
      "country": "gb",
      "published_at": "2026-05-09T09:00:00+00:00"
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fetcher_mediastack.py
import json
import os
from datetime import timezone
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.mediastack import MediaStackFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "mediastack_response.json").read_text()
)
BASE_URL = "http://api.mediastack.com/v1/news"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = MediaStackFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].outlet == "BBC News"
    assert articles[0].fetcher == "mediastack"
    assert articles[0].author == "John Smith"
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo is not None
    assert articles[1].image_url is None


@respx.mock
def test_mediastack_error_in_body():
    error_body = {"error": {"code": 104, "message": "User has reached their monthly request limit."}}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=error_body))
    fetcher = MediaStackFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_invalid_key_error():
    error_body = {"error": {"code": 101, "message": "Invalid API key."}}
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=error_body))
    fetcher = MediaStackFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_http_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = MediaStackFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("MEDIASTACK_API_KEY"),
    reason="MEDIASTACK_API_KEY not set",
)
def test_integration_live():
    fetcher = MediaStackFetcher(api_key=os.environ["MEDIASTACK_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "mediastack" for a in articles)
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_fetcher_mediastack.py -v -k "not integration"
```

Expected: `ImportError` or attribute errors.

- [ ] **Step 4: Implement `src/pocket_news/fetchers/mediastack.py`**

```python
"""MediaStack news fetcher."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "http://api.mediastack.com/v1/news"


class MediaStackFetcher(BaseFetcher):
    """Fetches articles from MediaStack API."""

    id: ClassVar[str] = "mediastack"
    display_name: ClassVar[str] = "MediaStack"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=lookback_days)
        date_range = f"{start.strftime('%Y-%m-%d')},{now.strftime('%Y-%m-%d')}"

        params = {
            "access_key": self.api_key,
            "keywords": topic,
            "languages": language,
            "date": date_range,
            "sort": "published_desc",
            "limit": max_articles,
        }

        try:
            response = self._client.get(BASE_URL, params=params)
        except Exception as exc:
            raise FetcherError(f"MediaStack request failed: {exc}") from exc

        if response.status_code >= 500:
            raise FetcherError(f"MediaStack server error: {response.status_code}")

        body = response.json()

        if "error" in body:
            code = body["error"].get("code", 0)
            message = body["error"].get("message", "Unknown error")
            if code == 101:
                raise ConfigurationError(f"MediaStack: {message}")
            if code == 104:
                raise RateLimitError(f"MediaStack: {message}")
            raise FetcherError(f"MediaStack error {code}: {message}")

        articles = []
        for item in body.get("data", []):
            outlet_name, domain = resolve_outlet(item.get("url", ""))
            source = item.get("source")
            if source:
                outlet_name = source

            published_at = None
            raw_date = item.get("published_at")
            if raw_date:
                try:
                    published_at = dateutil_parser.isoparse(raw_date).astimezone(timezone.utc)
                except Exception:
                    pass

            snippet = item.get("description")
            if snippet:
                snippet = _strip_html(snippet)

            articles.append(
                Article(
                    fetcher=self.id,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    author=item.get("author"),
                    outlet=outlet_name,
                    outlet_domain=domain,
                    language=item.get("language"),
                    country=item.get("country"),
                    categories=[item["category"]] if item.get("category") else [],
                    published_at=published_at,
                    image_url=item.get("image") or None,
                )
            )

        return articles
```

- [ ] **Step 5: Register in `src/pocket_news/fetchers/__init__.py`**

```python
from .base import BaseFetcher  # noqa: F401
from .mediastack import MediaStackFetcher

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {
    "mediastack": MediaStackFetcher,
}
```

- [ ] **Step 6: Run to verify tests pass**

```bash
pytest tests/test_fetcher_mediastack.py -v -k "not integration"
```

Expected: 4 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/mediastack_response.json tests/test_fetcher_mediastack.py src/pocket_news/fetchers/mediastack.py src/pocket_news/fetchers/__init__.py
git commit -m "feat: MediaStackFetcher"
```

---

## Task 8: The News API fetcher

**Files:**
- Create: `tests/fixtures/the_news_response.json`
- Create: `tests/test_fetcher_the_news.py`
- Modify: `src/pocket_news/fetchers/the_news.py`
- Modify: `src/pocket_news/fetchers/__init__.py`

- [ ] **Step 1: Create `tests/fixtures/the_news_response.json`**

```json
{
  "meta": {"found": 50, "returned": 2, "limit": 2, "page": 1},
  "data": [
    {
      "uuid": "6b100cae-f689-4b5e-8a7d-1234567890ab",
      "title": "EU AI Act: What businesses need to know",
      "description": "A comprehensive guide to EU AI Act requirements for businesses.",
      "keywords": "EU, AI Act, compliance",
      "snippet": "Businesses must comply by August 2026.",
      "url": "https://politico.com/news/eu-ai-act-business-guide",
      "image_url": "https://politico.com/images/ai-act.jpg",
      "language": "en",
      "published_at": "2026-05-10T02:24:31.000000Z",
      "source": "politico.com",
      "categories": ["politics", "technology"],
      "relevance_score": null,
      "locale": "us"
    },
    {
      "uuid": "abcdef12-3456-7890-abcd-ef1234567890",
      "title": "EU regulators issue first AI Act guidance",
      "description": null,
      "keywords": "EU, AI, regulation",
      "snippet": "Detailed guidance on implementing the AI Act.",
      "url": "https://ft.com/content/eu-ai-guidance-2026",
      "image_url": null,
      "language": "en",
      "published_at": "2026-05-09T18:00:00.000000Z",
      "source": "ft.com",
      "categories": ["technology"],
      "relevance_score": null,
      "locale": "gb"
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fetcher_the_news.py
import json
import os
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.the_news import TheNewsAPIFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "the_news_response.json").read_text()
)
BASE_URL = "https://api.thenewsapi.com/v1/news/all"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].fetcher == "the_news"
    assert articles[0].source_id == "6b100cae-f689-4b5e-8a7d-1234567890ab"
    assert articles[0].outlet == "Politico"
    assert articles[0].snippet is not None
    assert articles[1].image_url is None
    assert articles[1].country == "gb"


@respx.mock
def test_snippet_falls_back_to_snippet_field():
    fixture = json.loads(json.dumps(FIXTURE))
    fixture["data"][0]["description"] = None
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=fixture))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert articles[0].snippet == "Businesses must comply by August 2026."


@respx.mock
def test_401_raises_configuration_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(401))
    fetcher = TheNewsAPIFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_429_raises_rate_limit_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(429))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = TheNewsAPIFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("THE_NEWS_API_KEY"),
    reason="THE_NEWS_API_KEY not set",
)
def test_integration_live():
    fetcher = TheNewsAPIFetcher(api_key=os.environ["THE_NEWS_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "the_news" for a in articles)
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_fetcher_the_news.py -v -k "not integration"
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/pocket_news/fetchers/the_news.py`**

```python
"""The News API fetcher (thenewsapi.com)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "https://api.thenewsapi.com/v1/news/all"
MAX_PAGE_REQUESTS = 3


class TheNewsAPIFetcher(BaseFetcher):
    """Fetches articles from The News API."""

    id: ClassVar[str] = "the_news"
    display_name: ClassVar[str] = "The News API"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        start = datetime.now(timezone.utc) - timedelta(days=lookback_days)
        articles: list[Article] = []
        page = 1

        while len(articles) < max_articles and page <= MAX_PAGE_REQUESTS:
            params = {
                "api_token": self.api_key,
                "search": topic,
                "language": language,
                "published_after": start.strftime("%Y-%m-%dT%H:%M:%S"),
                "sort": "published_at",
                "limit": min(max_articles - len(articles), 3),
                "page": page,
            }

            try:
                response = self._client.get(BASE_URL, params=params)
            except Exception as exc:
                raise FetcherError(f"The News API request failed: {exc}") from exc

            if response.status_code == 401:
                raise ConfigurationError("The News API: invalid API key")
            if response.status_code in (402, 429):
                raise RateLimitError("The News API: rate limit or quota exceeded")
            if response.status_code >= 500:
                raise FetcherError(f"The News API server error: {response.status_code}")

            body = response.json()
            data = body.get("data", [])
            if not data:
                break

            for item in data:
                snippet = item.get("description") or item.get("snippet")
                if snippet:
                    snippet = _strip_html(snippet)

                outlet_name, domain = resolve_outlet(item.get("source", ""))

                published_at = None
                raw_date = item.get("published_at")
                if raw_date:
                    try:
                        published_at = dateutil_parser.isoparse(raw_date).astimezone(timezone.utc)
                    except Exception:
                        pass

                articles.append(
                    Article(
                        fetcher=self.id,
                        source_id=item.get("uuid"),
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        snippet=snippet,
                        outlet=outlet_name,
                        outlet_domain=domain,
                        language=item.get("language"),
                        country=item.get("locale"),
                        categories=item.get("categories") or [],
                        published_at=published_at,
                        image_url=item.get("image_url") or None,
                    )
                )

            meta = body.get("meta", {})
            if len(articles) >= meta.get("found", 0):
                break
            page += 1

        return articles
```

- [ ] **Step 5: Register in `src/pocket_news/fetchers/__init__.py`**

```python
from .base import BaseFetcher  # noqa: F401
from .mediastack import MediaStackFetcher
from .the_news import TheNewsAPIFetcher

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {
    "mediastack": MediaStackFetcher,
    "the_news": TheNewsAPIFetcher,
}
```

- [ ] **Step 6: Run to verify tests pass**

```bash
pytest tests/test_fetcher_the_news.py -v -k "not integration"
```

Expected: 5 passed.

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/the_news_response.json tests/test_fetcher_the_news.py src/pocket_news/fetchers/the_news.py src/pocket_news/fetchers/__init__.py
git commit -m "feat: TheNewsAPIFetcher"
```

---

## Task 9: World News API fetcher

**Files:**
- Create: `tests/fixtures/world_news_response.json`
- Create: `tests/test_fetcher_world_news.py`
- Modify: `src/pocket_news/fetchers/world_news.py`
- Modify: `src/pocket_news/fetchers/__init__.py`

- [ ] **Step 1: Create `tests/fixtures/world_news_response.json`**

```json
{
  "offset": 0,
  "number": 2,
  "available": 100,
  "news": [
    {
      "id": 123456789,
      "title": "EU AI Act enforcement begins",
      "text": "The European Union has begun enforcing the AI Act, marking a significant milestone in global AI regulation. The regulation requires companies to assess risks associated with their AI systems.",
      "summary": "The EU has started enforcing the AI Act.",
      "url": "https://reuters.com/technology/eu-ai-act-enforcement-2026-05-10",
      "image": "https://reuters.com/images/eu-ai.jpg",
      "publish_date": "2026-05-10 14:32:00",
      "author": "Jane Doe",
      "language": "en",
      "source_country": "us",
      "sentiment": 0.12,
      "categories": ["technology", "politics"]
    },
    {
      "id": 987654321,
      "title": "Companies scramble to comply with EU AI rules",
      "text": "Major technology companies are racing to comply with the new EU AI Act requirements ahead of the deadline.",
      "summary": null,
      "url": "https://nytimes.com/2026/05/10/technology/eu-ai-act-compliance.html",
      "image": null,
      "publish_date": "2026-05-10 10:00:00",
      "author": null,
      "language": "en",
      "source_country": "us",
      "sentiment": -0.05,
      "categories": ["technology"]
    }
  ]
}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/test_fetcher_world_news.py
import json
import os
from datetime import timezone
from pathlib import Path

import httpx
import pytest
import respx

from pocket_news.fetchers.world_news import WorldNewsFetcher
from pocket_news.exceptions import FetcherError, RateLimitError, ConfigurationError

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "world_news_response.json").read_text()
)
BASE_URL = "https://api.worldnewsapi.com/search-news"


@respx.mock
def test_fetch_returns_articles():
    respx.get(BASE_URL).mock(return_value=httpx.Response(200, json=FIXTURE))
    fetcher = WorldNewsFetcher(api_key="test-key")
    articles = fetcher.fetch("EU AI Act", max_articles=2, language="en", lookback_days=7)
    assert len(articles) == 2
    assert articles[0].fetcher == "world_news"
    assert articles[0].source_id == "123456789"
    assert articles[0].outlet == "Reuters"
    assert articles[0].content is not None  # World News returns full text
    assert articles[0].snippet == "The EU has started enforcing the AI Act."
    assert articles[0].author == "Jane Doe"
    assert articles[0].published_at is not None
    assert articles[0].published_at.tzinfo == timezone.utc
    assert articles[1].image_url is None
    assert articles[1].snippet is not None  # falls back to first 300 chars of text


@respx.mock
def test_401_raises_configuration_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(401))
    fetcher = WorldNewsFetcher(api_key="bad-key")
    with pytest.raises(ConfigurationError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_429_raises_rate_limit_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(429))
    fetcher = WorldNewsFetcher(api_key="test-key")
    with pytest.raises(RateLimitError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@respx.mock
def test_500_raises_fetcher_error():
    respx.get(BASE_URL).mock(return_value=httpx.Response(500))
    fetcher = WorldNewsFetcher(api_key="test-key")
    with pytest.raises(FetcherError):
        fetcher.fetch("test", max_articles=2, language="en", lookback_days=7)


@pytest.mark.skipif(
    not os.getenv("WORLD_NEWS_API_KEY"),
    reason="WORLD_NEWS_API_KEY not set",
)
def test_integration_live():
    fetcher = WorldNewsFetcher(api_key=os.environ["WORLD_NEWS_API_KEY"])
    articles = fetcher.fetch("technology", max_articles=2, language="en", lookback_days=7)
    assert len(articles) >= 1
    assert all(a.fetcher == "world_news" for a in articles)
```

- [ ] **Step 3: Run to verify it fails**

```bash
pytest tests/test_fetcher_world_news.py -v -k "not integration"
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/pocket_news/fetchers/world_news.py`**

```python
"""World News API fetcher (worldnewsapi.com)."""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import ClassVar

from dateutil import parser as dateutil_parser

from ..exceptions import ConfigurationError, FetcherError, RateLimitError
from ..models import Article
from .base import BaseFetcher, _strip_html, resolve_outlet

logger = logging.getLogger(__name__)

BASE_URL = "https://api.worldnewsapi.com/search-news"


class WorldNewsFetcher(BaseFetcher):
    """Fetches articles from World News API."""

    id: ClassVar[str] = "world_news"
    display_name: ClassVar[str] = "World News API"

    def fetch(
        self,
        topic: str,
        max_articles: int,
        language: str,
        lookback_days: int,
    ) -> list[Article]:
        earliest = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        params = {
            "text": topic,
            "language": language,
            "earliest-publish-date": earliest.strftime("%Y-%m-%dT%H:%M:%S"),
            "sort": "publish-time",
            "sort-direction": "DESC",
            "number": max_articles,
        }
        headers = {"x-api-key": self.api_key}

        try:
            response = self._client.get(BASE_URL, params=params, headers=headers)
        except Exception as exc:
            raise FetcherError(f"World News API request failed: {exc}") from exc

        if response.status_code == 401:
            raise ConfigurationError("World News API: invalid API key")
        if response.status_code in (402, 429):
            raise RateLimitError("World News API: rate limit or quota exceeded")
        if response.status_code >= 500:
            raise FetcherError(f"World News API server error: {response.status_code}")

        body = response.json()
        articles = []

        for item in body.get("news", []):
            outlet_name, domain = resolve_outlet(item.get("url", ""))

            raw_text = item.get("text", "")
            content = _strip_html(raw_text) if raw_text else None

            raw_summary = item.get("summary")
            if raw_summary:
                snippet = _strip_html(raw_summary)
            elif content:
                snippet = content[:300]
            else:
                snippet = None

            published_at = None
            raw_date = item.get("publish_date")
            if raw_date:
                try:
                    published_at = dateutil_parser.parse(raw_date).replace(tzinfo=timezone.utc)
                except Exception:
                    pass

            articles.append(
                Article(
                    fetcher=self.id,
                    source_id=str(item["id"]) if item.get("id") is not None else None,
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=snippet,
                    content=content,
                    author=item.get("author") or None,
                    outlet=outlet_name,
                    outlet_domain=domain,
                    language=item.get("language"),
                    country=item.get("source_country"),
                    categories=item.get("categories") or [],
                    published_at=published_at,
                    image_url=item.get("image") or None,
                )
            )

        return articles
```

- [ ] **Step 5: Register in `src/pocket_news/fetchers/__init__.py`**

```python
from .base import BaseFetcher  # noqa: F401
from .mediastack import MediaStackFetcher
from .the_news import TheNewsAPIFetcher
from .world_news import WorldNewsFetcher

FETCHER_REGISTRY: dict[str, type[BaseFetcher]] = {
    "mediastack": MediaStackFetcher,
    "the_news": TheNewsAPIFetcher,
    "world_news": WorldNewsFetcher,
}
```

- [ ] **Step 6: Run to verify tests pass**

```bash
pytest tests/test_fetcher_world_news.py -v -k "not integration"
```

Expected: 4 passed.

- [ ] **Step 7: Run the full suite to catch regressions**

```bash
pytest tests/ -v -k "not integration"
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add tests/fixtures/world_news_response.json tests/test_fetcher_world_news.py src/pocket_news/fetchers/world_news.py src/pocket_news/fetchers/__init__.py
git commit -m "feat: WorldNewsFetcher — all three fetchers implemented"
```

---

## Task 10: Deduplication

**Files:**
- Create: `src/pocket_news/dedupe.py`
- Create: `tests/test_dedupe.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dedupe.py
from pocket_news.dedupe import ArticleDeduplicator
from pocket_news.models import Article


def _art(fetcher="world_news", title="Title", url="https://example.com", content=None, snippet=None):
    return Article(fetcher=fetcher, title=title, url=url, outlet="Example", content=content, snippet=snippet)


def test_empty_list():
    assert ArticleDeduplicator.deduplicate([]) == []


def test_no_duplicates_unchanged():
    articles = [_art(url="https://a.com", title="Article A"), _art(url="https://b.com", title="Article B")]
    assert len(ArticleDeduplicator.deduplicate(articles)) == 2


def test_exact_url_dedup():
    a1 = _art(url="https://example.com/article?utm_source=tw", title="T1")
    a2 = _art(url="https://example.com/article?ref=fb", title="T1")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_url_trailing_slash_dedup():
    a1 = _art(url="https://example.com/article/")
    a2 = _art(url="https://example.com/article")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_exact_title_dedup():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins")
    a2 = _art(url="https://b.com", title="EU AI Act enforcement begins")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_similar_title_dedup():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins across Europe")
    a2 = _art(url="https://b.com", title="EU AI Act enforcement begins in Europe")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 1


def test_dissimilar_titles_kept():
    a1 = _art(url="https://a.com", title="EU AI Act enforcement begins")
    a2 = _art(url="https://b.com", title="SpaceX launches Starship")
    assert len(ArticleDeduplicator.deduplicate([a1, a2])) == 2


def test_prefers_content_over_snippet():
    with_content = _art(url="https://a.com", title="Same Title", content="Full article body.")
    snippet_only = _art(url="https://b.com", title="Same Title", snippet="Short snippet.")
    result = ArticleDeduplicator.deduplicate([snippet_only, with_content])
    assert len(result) == 1
    assert result[0].content == "Full article body."


def test_prefers_longer_content():
    short = _art(url="https://a.com", title="Same", content="Short.")
    long = _art(url="https://b.com", title="Same", content="A much longer article body with more details.")
    result = ArticleDeduplicator.deduplicate([short, long])
    assert len(result) == 1
    assert "longer" in result[0].content
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_dedupe.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/dedupe.py`**

```python
"""Article deduplication â€” URL canonicalization + title similarity."""
from __future__ import annotations
import difflib
from urllib.parse import urlparse, urlunparse
from .models import Article

_SIMILARITY_THRESHOLD = 0.85


def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
        norm = p._replace(scheme=p.scheme.lower(), netloc=p.netloc.lower(), query="", fragment="")
        return urlunparse(norm._replace(path=norm.path.rstrip("/")))
    except Exception:
        return url.lower()


def _normalize_title(title: str) -> str:
    return " ".join(title.lower().split())


def _prefer(a: Article, b: Article) -> Article:
    a_has = bool(a.content)
    b_has = bool(b.content)
    if a_has != b_has:
        return a if a_has else b
    return a if len(a.content or a.snippet or "") >= len(b.content or b.snippet or "") else b


class ArticleDeduplicator:
    """Deduplicates a list of Articles by URL canonicalization and title similarity."""

    @staticmethod
    def deduplicate(articles: list[Article]) -> list[Article]:
        kept: list[Article] = []
        canonical_urls: set[str] = set()
        norm_titles: list[str] = []

        for article in articles:
            canon = _canonical_url(article.url)
            norm = _normalize_title(article.title)

            if canon in canonical_urls:
                idx = next((i for i, a in enumerate(kept) if _canonical_url(a.url) == canon), None)
                if idx is not None:
                    kept[idx] = _prefer(kept[idx], article)
                continue

            matched = None
            for i, t in enumerate(norm_titles):
                if t == norm or difflib.SequenceMatcher(None, norm, t).ratio() >= _SIMILARITY_THRESHOLD:
                    matched = i
                    break

            if matched is not None:
                kept[matched] = _prefer(kept[matched], article)
                continue

            canonical_urls.add(canon)
            norm_titles.append(norm)
            kept.append(article)

        return kept
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_dedupe.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/dedupe.py tests/test_dedupe.py
git commit -m "feat: ArticleDeduplicator â€” URL canonicalization + title similarity"
```

---

## Task 11: Image fetching

**Files:**
- Create: `src/pocket_news/images.py`
- Create: `tests/test_images.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_images.py
import base64, httpx, pytest, respx
from pocket_news.images import fetch_and_encode_image

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


@respx.mock
def test_successful_jpeg():
    respx.get("https://example.com/img.jpg").mock(
        return_value=httpx.Response(200, content=JPEG, headers={"content-type": "image/jpeg"})
    )
    b64, mime = fetch_and_encode_image("https://example.com/img.jpg")
    assert mime == "image/jpeg"
    assert base64.b64decode(b64) == JPEG


@respx.mock
def test_404_returns_none():
    respx.get("https://example.com/missing.jpg").mock(return_value=httpx.Response(404))
    assert fetch_and_encode_image("https://example.com/missing.jpg") == (None, None)


@respx.mock
def test_non_image_content_type():
    respx.get("https://example.com/page").mock(
        return_value=httpx.Response(200, content=b"<html>", headers={"content-type": "text/html"})
    )
    assert fetch_and_encode_image("https://example.com/page") == (None, None)


@respx.mock
def test_oversized_returns_none():
    respx.get("https://example.com/large.jpg").mock(
        return_value=httpx.Response(200, content=b"x" * 3_000_000, headers={"content-type": "image/jpeg"})
    )
    assert fetch_and_encode_image("https://example.com/large.jpg", max_bytes=2_000_000) == (None, None)


def test_network_error_returns_none():
    assert fetch_and_encode_image("https://no-such-host-xyz.invalid/img.jpg", timeout=0.001) == (None, None)
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_images.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/images.py`**

```python
"""Image fetching and base64 encoding."""
from __future__ import annotations
import base64, logging, mimetypes
import httpx

logger = logging.getLogger(__name__)


def fetch_and_encode_image(
    url: str,
    max_bytes: int = 2_000_000,
    timeout: float = 5.0,
) -> tuple[str | None, str | None]:
    """Returns (base64_data, mime_type) or (None, None) on any failure. Never raises."""
    try:
        with httpx.Client(timeout=timeout) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                content_type = response.headers.get("content-type", "")
                mime_type = content_type.split(";")[0].strip()
                if not mime_type.startswith("image/"):
                    guessed, _ = mimetypes.guess_type(url)
                    if guessed and guessed.startswith("image/"):
                        mime_type = guessed
                    else:
                        return None, None
                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes(chunk_size=8192):
                    total += len(chunk)
                    if total > max_bytes:
                        return None, None
                    chunks.append(chunk)
                return base64.b64encode(b"".join(chunks)).decode("ascii"), mime_type
    except Exception as exc:
        logger.debug("Image fetch failed for %s: %s", url, exc)
        return None, None
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_images.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/images.py tests/test_images.py
git commit -m "feat: fetch_and_encode_image â€” streaming, size-limited, never raises"
```

---

## Task 12: Cache

**Files:**
- Create: `src/pocket_news/cache.py`
- Create: `tests/test_cache.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cache.py
import time, pytest
from datetime import datetime, timezone
from pocket_news.cache import CacheStore
from pocket_news.models import SynthesizedArticle


def test_set_and_get_roundtrip(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "EU AI Act", "en", "standard", "qwen3:14b")
    result = cache.get("EU AI Act", "en", "standard", "qwen3:14b")
    assert result is not None
    assert result.headline == sample_synthesized_article.headline


def test_cache_miss_returns_none(tmp_path):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    assert cache.get("unknown topic", "en", "standard", "qwen3:14b") is None


def test_stale_file_is_miss(tmp_path, sample_synthesized_article, monkeypatch):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=1)
    cache.set(sample_synthesized_article, "topic", "en", "standard", "qwen3:14b")
    orig = time.time
    monkeypatch.setattr(time, "time", lambda: orig() + 120)
    assert cache.get("topic", "en", "standard", "qwen3:14b") is None


def test_no_results_not_cached(tmp_path):
    no_results = SynthesizedArticle(
        status="no_results", topic="x", output_language="en",
        headline="", lead="", body="", key_points=[],
        model="qwen3:14b", generated_at=datetime(2026, 5, 11, tzinfo=timezone.utc),
        article_count=0, fetcher_status={},
    )
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(no_results, "x", "en", "standard", "qwen3:14b")
    assert cache.get("x", "en", "standard", "qwen3:14b") is None


def test_clear_returns_count(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "topic1", "en", "standard", "qwen3:14b")
    cache.set(sample_synthesized_article, "topic2", "en", "standard", "qwen3:14b")
    assert cache.clear() == 2
    assert list(tmp_path.glob("*.json")) == []


def test_key_includes_language_and_length(tmp_path, sample_synthesized_article):
    cache = CacheStore(cache_dir=tmp_path, ttl_minutes=60)
    cache.set(sample_synthesized_article, "topic", "en", "standard", "qwen3:14b")
    assert cache.get("topic", "es", "standard", "qwen3:14b") is None
    assert cache.get("topic", "en", "brief", "qwen3:14b") is None
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_cache.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/cache.py`**

```python
"""Local file cache for SynthesizedArticle results."""
from __future__ import annotations
import hashlib, logging, time
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
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_cache.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/cache.py tests/test_cache.py
git commit -m "feat: CacheStore â€” TTL file cache, no-results skip"
```

---

## Task 13: Prompts

**Files:**
- Create: `src/pocket_news/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_prompts.py
from datetime import datetime, timezone
from pocket_news.models import Article
from pocket_news.prompts import build_synthesis_prompt, LENGTH_TO_WORD_COUNT


def _art(title, outlet, snippet, url="https://example.com"):
    return Article(fetcher="world_news", title=title, url=url, outlet=outlet, snippet=snippet,
                   published_at=datetime(2026, 5, 10, tzinfo=timezone.utc))


def test_length_presets():
    assert LENGTH_TO_WORD_COUNT == {"brief": 300, "standard": 600, "detailed": 1000}


def test_prompt_contains_article_content(three_articles):
    _, user_p = build_synthesis_prompt("EU AI Act", three_articles, "English", 600, "neutral")
    assert "EU AI Act" in user_p
    assert "Reuters" in user_p
    assert "Politico" in user_p
    assert "BBC News" in user_p
    assert "600" in user_p


def test_language_mentioned_multiple_times():
    articles = [_art("Test", "Example", "snippet")]
    _, user_p = build_synthesis_prompt("test", articles, "Spanish", 300, "neutral")
    assert user_p.count("Spanish") >= 3


def test_section_markers_present(three_articles):
    _, user_p = build_synthesis_prompt("test", three_articles, "English", 600, "neutral")
    for marker in ["===HEADLINE===", "===LEAD===", "===BODY===", "===KEY_POINTS==="]:
        assert marker in user_p


def test_articles_block_bounded():
    long_content = "x" * 5000
    articles = [Article(fetcher="world_news", title=f"A{i}", url=f"https://a{i}.com",
                        outlet="Example", content=long_content) for i in range(5)]
    _, user_p = build_synthesis_prompt("test", articles, "English", 600, "neutral")
    assert len(user_p) < 20000
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_prompts.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/prompts.py`**

```python
"""Prompt templates and builder for OllamaSynthesizer."""
from __future__ import annotations
from .models import Article

LENGTH_TO_WORD_COUNT: dict[str, int] = {"brief": 300, "standard": 600, "detailed": 1000}

SYSTEM_PROMPT = """You are a professional news editor. Given a set of articles from different outlets covering the same topic, your job is to synthesize them into a single coherent news article that reads as one cohesive piece â€” not a list of summaries.

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
    return f"[{n}] {article.outlet} â€” {date_str} â€” {article.url}\nTITLE: {article.title}\n{body}\n---"


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
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_prompts.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/pocket_news/prompts.py tests/test_prompts.py
git commit -m "feat: prompt templates and build_synthesis_prompt()"
```

---

## Task 14: Synthesizer

**Files:**
- Create: `src/pocket_news/synthesizer.py`
- Create: `tests/test_parser.py`
- Create: `tests/test_synthesizer_unit.py`

- [ ] **Step 1: Write `tests/test_parser.py`**

```python
# tests/test_parser.py
import pytest
from pocket_news.synthesizer import _parse_synthesis_output, _strip_think_blocks
from pocket_news.exceptions import SynthesisParseError

WELL_FORMED = """===HEADLINE===
EU AI Act Enforcement Underway

===LEAD===
The European Union has begun enforcing its landmark AI Act.

===BODY===
The EU AI Act entered enforcement phase this week, according to Reuters.

===KEY_POINTS===
- EU AI Act enforcement launched
- Companies face August 2026 deadline
- High-risk systems require full risk assessment

===SOURCES===
- Reuters
"""


def test_parse_well_formed():
    result = _parse_synthesis_output(WELL_FORMED)
    assert result["headline"] == "EU AI Act Enforcement Underway"
    assert "European Union" in result["lead"]
    assert len(result["key_points"]) == 3


def test_strip_think_blocks():
    raw = "<think>reasoning...</think>\n\n===HEADLINE===\nReal Headline"
    assert "<think>" not in _strip_think_blocks(raw)
    assert "Real Headline" in _strip_think_blocks(raw)


def test_think_stripped_before_parse():
    result = _parse_synthesis_output("<think>reasoning</think>\n" + WELL_FORMED)
    assert result["headline"] == "EU AI Act Enforcement Underway"


def test_quoted_headline_unquoted():
    raw = WELL_FORMED.replace("EU AI Act Enforcement Underway", '"EU AI Act Enforcement Underway"')
    assert _parse_synthesis_output(raw)["headline"] == "EU AI Act Enforcement Underway"


def test_numbered_key_points_stripped():
    raw = WELL_FORMED.replace(
        "- EU AI Act enforcement launched\n- Companies face August 2026 deadline\n- High-risk systems require full risk assessment",
        "1. EU AI Act enforcement launched\n2. Companies face August 2026 deadline\nâ€¢ High-risk systems require full risk assessment",
    )
    result = _parse_synthesis_output(raw)
    assert result["key_points"][0] == "EU AI Act enforcement launched"
    assert result["key_points"][2] == "High-risk systems require full risk assessment"


def test_missing_section_raises():
    no_body = "===HEADLINE===\nHeadline\n\n===LEAD===\nLead.\n\n===KEY_POINTS===\n- Point\n"
    with pytest.raises(SynthesisParseError, match="body"):
        _parse_synthesis_output(no_body)
```

- [ ] **Step 2: Write `tests/test_synthesizer_unit.py`**

```python
# tests/test_synthesizer_unit.py
import httpx, pytest
from unittest.mock import MagicMock, patch
from pocket_news.synthesizer import OllamaSynthesizer
from pocket_news.exceptions import OllamaUnavailableError

MOCK_TEXT = """===HEADLINE===
EU AI Act Enforcement Underway

===LEAD===
The EU has launched enforcement.

===BODY===
The EU AI Act enforcement began, according to Reuters.

===KEY_POINTS===
- EU AI Act launched
- August 2026 deadline

===SOURCES===
- Reuters
"""


def _mock_response(text):
    msg = MagicMock(); msg.content = text
    resp = MagicMock(); resp.message = msg
    return resp


def test_no_think_appended_for_qwen3(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b", disable_thinking_mode=True)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert user_msg["content"].endswith("/no_think")


def test_no_think_not_appended_for_gemma(three_articles):
    synth = OllamaSynthesizer(model="gemma4:latest", disable_thinking_mode=True)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert not user_msg["content"].endswith("/no_think")


def test_no_think_not_appended_when_disabled_false(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b", disable_thinking_mode=False)
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        msgs = MC.return_value.chat.call_args.kwargs["messages"]
        user_msg = next(m for m in msgs if m["role"] == "user")
        assert not user_msg["content"].endswith("/no_think")


def test_synthesize_returns_parsed_dict(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b")
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.return_value = _mock_response(MOCK_TEXT)
        result = synth.synthesize("EU AI Act", three_articles, "English", 600, "neutral")
        assert result["headline"] == "EU AI Act Enforcement Underway"
        assert len(result["key_points"]) == 2


def test_connect_error_raises_ollama_unavailable(three_articles):
    synth = OllamaSynthesizer(model="qwen3:14b")
    with patch("pocket_news.synthesizer.Client") as MC:
        MC.return_value.chat.side_effect = httpx.ConnectError("refused")
        with pytest.raises(OllamaUnavailableError, match="ollama serve"):
            synth.synthesize("topic", three_articles, "English", 600, "neutral")
```

- [ ] **Step 3: Run to verify both fail**

```bash
pytest tests/test_parser.py tests/test_synthesizer_unit.py -v
```

Expected: `ImportError`.

- [ ] **Step 4: Implement `src/pocket_news/synthesizer.py`**

```python
"""OllamaSynthesizer â€” wraps the Ollama client for news synthesis."""
from __future__ import annotations
import logging, re
import httpx
from ollama import Client
from .exceptions import OllamaUnavailableError, SynthesisParseError
from .models import Article
from .prompts import build_synthesis_prompt

logger = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"===\s*([A-Z_]+)\s*===")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_BULLET_PREFIX = re.compile(r"^[\-\*â€¢]|\d+[\.\)]\s*")


def _is_qwen3(model: str) -> bool:
    return model.lower().startswith("qwen3")


def _strip_think_blocks(text: str) -> str:
    return _THINK_PATTERN.sub("", text).strip()


def _maybe_append_no_think(prompt: str, model: str, disable: bool) -> str:
    if disable and _is_qwen3(model):
        return prompt.rstrip() + "\n\n/no_think"
    return prompt


def _parse_key_points(raw: str) -> list[str]:
    points = []
    for line in raw.splitlines():
        line = _BULLET_PREFIX.sub("", line.strip()).strip()
        if line:
            points.append(line)
    return points[:7]


def _parse_synthesis_output(raw: str) -> dict:
    raw = _strip_think_blocks(raw)
    parts = SECTION_PATTERN.split(raw)
    sections: dict[str, str] = {}
    for i in range(1, len(parts), 2):
        name = parts[i].strip().lower()
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        sections[name] = content

    missing = {"headline", "lead", "body", "key_points"} - sections.keys()
    if missing:
        raise SynthesisParseError(
            f"Missing required sections: {missing}. Got: {list(sections.keys())}. "
            f"Raw (truncated): {raw[:2000]}"
        )

    return {
        "headline": sections["headline"].strip().strip('"').strip("'"),
        "lead": sections["lead"],
        "body": sections["body"],
        "key_points": _parse_key_points(sections["key_points"]),
    }


class OllamaSynthesizer:
    """Calls a local Ollama model to synthesize articles into one piece."""

    def __init__(
        self,
        model: str = "qwen3:14b",
        host: str = "http://localhost:11434",
        timeout_seconds: int = 120,
        disable_thinking_mode: bool = True,
    ) -> None:
        self.model = model
        self.host = host
        self.timeout = timeout_seconds
        self.disable_thinking_mode = disable_thinking_mode

    def synthesize(
        self,
        topic: str,
        articles: list[Article],
        output_language_name: str,
        target_word_count: int,
        tone: str = "neutral",
    ) -> dict:
        """Returns parsed dict: headline, lead, body, key_points."""
        system_prompt, user_prompt = build_synthesis_prompt(
            topic, articles, output_language_name, target_word_count, tone
        )
        user_prompt = _maybe_append_no_think(user_prompt, self.model, self.disable_thinking_mode)
        client = Client(host=self.host)
        try:
            response = client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.3, "top_p": 0.9, "num_ctx": 8192},
            )
        except httpx.ConnectError as exc:
            raise OllamaUnavailableError(
                f"Could not reach Ollama at {self.host}. Is 'ollama serve' running?"
            ) from exc
        except Exception as exc:
            msg = str(exc).lower()
            if "not found" in msg or "pull" in msg:
                raise OllamaUnavailableError(
                    f"Model '{self.model}' is not available. Run 'ollama pull {self.model}'."
                ) from exc
            raise
        return _parse_synthesis_output(response.message.content)

    def healthcheck(self) -> bool:
        try:
            models_response = Client(host=self.host).list()
            available = {m.model.split(":")[0] for m in models_response.models}
            return self.model.split(":")[0] in available
        except Exception:
            return False
```

- [ ] **Step 5: Run to verify tests pass**

```bash
pytest tests/test_parser.py tests/test_synthesizer_unit.py -v
```

Expected: 11 passed.

- [ ] **Step 6: Commit**

```bash
git add src/pocket_news/synthesizer.py tests/test_parser.py tests/test_synthesizer_unit.py
git commit -m "feat: OllamaSynthesizer â€” parse, no_think directive, Ollama error handling"
```

---

## Task 15: Agent orchestrator

**Files:**
- Create: `src/pocket_news/agent.py`
- Create: `tests/test_agent.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
import pytest
from pocket_news.agent import NewsAgent
from pocket_news.config import AgentConfig
from pocket_news.models import Article
from pocket_news.exceptions import OllamaUnavailableError


def _config(**kw):
    defaults = dict(world_news_api_key="k1", the_news_api_key=None, mediastack_api_key=None,
                    cache_enabled=False, fetch_image_data=False)
    defaults.update(kw)
    return AgentConfig(**defaults)


def _article(fetcher="world_news"):
    return Article(fetcher=fetcher, title=f"Title", url=f"https://{fetcher}.com",
                   outlet="Example", snippet="snippet.",
                   published_at=datetime(2026, 5, 10, tzinfo=timezone.utc))


PARSED = {"headline": "Headline", "lead": "Lead.", "body": "Body.", "key_points": ["P1"]}


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_research_returns_synthesized_article(MockReg, MockSynth):
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = [_article()]
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)
    MockSynth.return_value.synthesize.return_value = PARSED

    agent = NewsAgent(config=_config())
    result = agent.research("EU AI Act")
    assert result.headline == "Headline"
    assert result.topic == "EU AI Act"
    assert result.output_language == "en"
    assert result.status in ("ok", "partial")


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_all_empty_returns_no_results(MockReg, MockSynth):
    mock_fetcher = MagicMock()
    mock_fetcher.fetch.return_value = []
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)

    agent = NewsAgent(config=_config())
    result = agent.research("obscure topic")
    assert result.status == "no_results"
    assert result.headline == ""
    MockSynth.return_value.synthesize.assert_not_called()


@patch("pocket_news.agent.OllamaSynthesizer")
@patch("pocket_news.agent.FETCHER_REGISTRY")
def test_cache_hit_skips_fetch(MockReg, MockSynth, tmp_path, sample_synthesized_article):
    from pocket_news.cache import CacheStore
    store = CacheStore(tmp_path, 60)
    store.set(sample_synthesized_article, "EU AI Act enforcement", "en", "standard", "qwen3:14b")

    mock_fetcher = MagicMock()
    MockReg.__getitem__ = MagicMock(return_value=lambda key, timeout: mock_fetcher)
    MockReg.__contains__ = MagicMock(return_value=True)

    config = AgentConfig(world_news_api_key="k1", the_news_api_key=None, mediastack_api_key=None,
                         cache_enabled=True, cache_dir=tmp_path, cache_ttl_minutes=60, fetch_image_data=False)
    agent = NewsAgent(config=config)
    result = agent.research("EU AI Act enforcement")
    assert result.headline == sample_synthesized_article.headline
    mock_fetcher.fetch.assert_not_called()
    MockSynth.return_value.synthesize.assert_not_called()
```

- [ ] **Step 2: Run to verify it fails**

```bash
pytest tests/test_agent.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement `src/pocket_news/agent.py`**

```python
"""NewsAgent â€” orchestrates fetch, dedupe, synthesize, cache."""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .cache import CacheStore
from .config import AgentConfig
from .dedupe import ArticleDeduplicator
from .exceptions import FetcherError, PocketNewsError
from .fetchers import FETCHER_REGISTRY
from .images import fetch_and_encode_image
from .languages import normalize_language
from .models import Article, SourceCitation, SynthesizedArticle
from .prompts import LENGTH_TO_WORD_COUNT
from .synthesizer import OllamaSynthesizer

logger = logging.getLogger(__name__)


class NewsAgent:
    """Fetches, deduplicates, and synthesizes news for a given topic."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self._config = config or AgentConfig()
        self._synthesizer: Optional[OllamaSynthesizer] = None
        self._cache: Optional[CacheStore] = None

    def _get_synthesizer(self) -> OllamaSynthesizer:
        if self._synthesizer is None:
            self._synthesizer = OllamaSynthesizer(
                model=self._config.ollama_model,
                host=self._config.ollama_host,
                timeout_seconds=self._config.ollama_timeout_seconds,
                disable_thinking_mode=self._config.disable_thinking_mode,
            )
        return self._synthesizer

    def _get_cache(self) -> Optional[CacheStore]:
        if not self._config.cache_enabled:
            return None
        if self._cache is None:
            cache_dir = Path(self._config.cache_dir) if self._config.cache_dir else None
            self._cache = CacheStore(cache_dir, self._config.cache_ttl_minutes)
        return self._cache

    def _fetch_all(self, topic: str) -> tuple[list[Article], dict[str, str]]:
        fetcher_pairs = self._config.resolved_fetchers()
        fetcher_status: dict[str, str] = {
            fid: "disabled"
            for fid in FETCHER_REGISTRY
            if not any(fid == p[0] for p in fetcher_pairs)
        }
        if not fetcher_pairs:
            return [], fetcher_status

        def _fetch_one(fetcher_id: str, api_key: str) -> tuple[str, list[Article], str]:
            fetcher_cls = FETCHER_REGISTRY[fetcher_id]
            fetcher = fetcher_cls(api_key, timeout=self._config.fetch_timeout_seconds)
            try:
                arts = fetcher.fetch(topic, self._config.max_articles_per_source, "en", self._config.lookback_days)
                return fetcher_id, arts, "ok" if arts else "empty"
            except FetcherError as exc:
                logger.warning("Fetcher %s failed: %s", fetcher_id, exc)
                return fetcher_id, [], f"failed: {exc}"
            finally:
                fetcher.close()

        all_articles: list[Article] = []
        with ThreadPoolExecutor(max_workers=len(fetcher_pairs)) as pool:
            for fid, arts, status in [f.result() for f in as_completed(
                [pool.submit(_fetch_one, fid, key) for fid, key in fetcher_pairs]
            )]:
                fetcher_status[fid] = status
                all_articles.extend(arts)

        return all_articles, fetcher_status

    def research(self, topic: str, language: str = "en", length: str = "standard") -> SynthesizedArticle:
        """Fetch, deduplicate, and synthesize articles for a topic."""
        iso_code, display_name = normalize_language(language)
        word_count = LENGTH_TO_WORD_COUNT.get(length, LENGTH_TO_WORD_COUNT["standard"])

        cache = self._get_cache()
        if cache:
            cached = cache.get(topic, iso_code, length, self._config.ollama_model)
            if cached:
                return cached

        raw_articles, fetcher_status = self._fetch_all(topic)

        if not raw_articles:
            return SynthesizedArticle(
                status="no_results", topic=topic, output_language=iso_code,
                headline="", lead="", body="", key_points=[],
                model=self._config.ollama_model, generated_at=datetime.now(timezone.utc),
                article_count=0, fetcher_status=fetcher_status,
            )

        articles = ArticleDeduplicator.deduplicate(raw_articles)

        featured_image_url = featured_image_b64 = featured_image_mime = None
        for art in articles:
            if art.image_url:
                featured_image_url = art.image_url
                if self._config.fetch_image_data:
                    featured_image_b64, featured_image_mime = fetch_and_encode_image(
                        art.image_url, self._config.image_max_bytes, self._config.image_fetch_timeout
                    )
                break

        parsed = self._get_synthesizer().synthesize(topic, articles, display_name, word_count, self._config.tone)
        status = "partial" if any(s.startswith("failed:") for s in fetcher_status.values()) else "ok"

        result = SynthesizedArticle(
            status=status, topic=topic, output_language=iso_code,
            headline=parsed["headline"], lead=parsed["lead"], body=parsed["body"],
            key_points=parsed["key_points"],
            featured_image_url=featured_image_url, featured_image_b64=featured_image_b64,
            featured_image_mime=featured_image_mime,
            sources=[SourceCitation(outlet=a.outlet, title=a.title, url=a.url,
                                    published_at=a.published_at, fetcher=a.fetcher) for a in articles],
            model=self._config.ollama_model, generated_at=datetime.now(timezone.utc),
            article_count=len(articles), fetcher_status=fetcher_status,
        )
        if cache:
            cache.set(result, topic, iso_code, length, self._config.ollama_model)
        return result

    def research_batch(self, topics: list[str], language: str = "en", length: str = "standard",
                       max_workers: int = 3) -> list[SynthesizedArticle]:
        """Runs research() for each topic concurrently. Preserves input order."""
        results: dict[int, SynthesizedArticle] = {}

        def _do(idx: int, topic: str) -> tuple[int, SynthesizedArticle]:
            try:
                return idx, self.research(topic, language=language, length=length)
            except PocketNewsError as exc:
                logger.warning("research() failed for %r: %s", topic, exc)
                iso_code, _ = normalize_language(language)
                return idx, SynthesizedArticle(
                    status="no_results", topic=topic, output_language=iso_code,
                    headline="", lead="", body="", key_points=[],
                    model=self._config.ollama_model, generated_at=datetime.now(timezone.utc),
                    article_count=0, fetcher_status={},
                )

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for idx, article in [f.result() for f in as_completed(
                [pool.submit(_do, i, t) for i, t in enumerate(topics)]
            )]:
                results[idx] = article

        return [results[i] for i in range(len(topics))]

    def clear_cache(self) -> int:
        cache = self._get_cache()
        return cache.clear() if cache else 0
```

- [ ] **Step 4: Run to verify tests pass**

```bash
pytest tests/test_agent.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Run full suite**

```bash
pytest tests/ -v -k "not integration"
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pocket_news/agent.py tests/test_agent.py
git commit -m "feat: NewsAgent â€” parallel fetch, dedupe, synthesize, cache, batch"
```

---

## Task 16: CLI

**Files:**
- Modify: `src/pocket_news/cli.py`

- [ ] **Step 1: Implement `src/pocket_news/cli.py`**

```python
"""Command-line interface for pocket-news."""
from __future__ import annotations
import argparse, logging, pathlib, sys


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
        '"""pocket-news API keys â€” NOT checked into git."""\n'
        f'WORLD_NEWS_API_KEY = "{world}"\n'
        f'THE_NEWS_API_KEY = "{the_news}"\n'
        f'MEDIASTACK_API_KEY = "{mediastack}"\n'
    )
    secrets_path.write_text(content, encoding="utf-8")
    print(f"\nKeys saved to {secrets_path}")
    print('Run: python -m pocket_news "your topic here"')


def _format_article(article) -> str:
    lines = [f"\n{'=' * 70}", article.headline, f"{'=' * 70}", "", article.lead, "", article.body,
             "", "KEY POINTS:"]
    for pt in article.key_points:
        lines.append(f"  â€¢ {pt}")
    lines += ["", "SOURCES:"]
    for src in article.sources:
        lines.append(f"  [{src.fetcher}] {src.outlet}: {src.url}")
    lines.append(f"\nStatus: {article.status} | Model: {article.model} | Articles: {article.article_count}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(prog="pocket-news", description="Fetch and synthesize news.")
    parser.add_argument("topic", nargs="?", help="Topic to research")
    parser.add_argument("--language", default="en")
    parser.add_argument("--length", choices=["brief", "standard", "detailed"], default="standard")
    parser.add_argument("--model", help="Ollama model override")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--no-cache", action="store_true")
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

    kwargs = {}
    if args.model:
        kwargs["ollama_model"] = args.model
    if args.no_cache:
        kwargs["cache_enabled"] = False

    agent = NewsAgent(config=AgentConfig(**kwargs) if kwargs else AgentConfig())
    article = agent.research(args.topic, language=args.language, length=args.length)

    if args.as_json:
        print(article.model_dump_json(indent=2))
    else:
        print(_format_article(article))
```

- [ ] **Step 2: Smoke-test**

```bash
python -m pocket_news --help
```

Expected: help text listing `topic`, `--language`, `--length`, `--model`, `--json`, `--no-cache`, `--setup`.

- [ ] **Step 3: Commit**

```bash
git add src/pocket_news/cli.py
git commit -m "feat: CLI with --setup interactive key configuration"
```

---

## Task 17: Examples

**Files:**
- Create: `examples/basic_usage.py`
- Create: `examples/translation.py`
- Create: `examples/multi_interest.py`

- [ ] **Step 1: Create `examples/basic_usage.py`**

```python
"""Basic usage â€” mirrors the README Quick Start."""
from pocket_news import NewsAgent

agent = NewsAgent()
article = agent.research("EU AI Act enforcement")

print(article.headline)
print(article.lead)
print(article.body)
for src in article.sources:
    print(f"- {src.outlet}: {src.url}")
```

- [ ] **Step 2: Create `examples/translation.py`**

```python
"""Synthesize the same topic in two languages."""
from pocket_news import NewsAgent

agent = NewsAgent()
topic = "EU AI Act enforcement"

for lang in ["Spanish", "French"]:
    article = agent.research(topic, language=lang, length="brief")
    print(f"\n=== {lang} ({article.output_language}) ===")
    print(article.headline)
    print(article.lead)
```

- [ ] **Step 3: Create `examples/multi_interest.py`**

```python
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
```

- [ ] **Step 4: Commit**

```bash
git add examples/
git commit -m "feat: examples â€” basic_usage, translation, multi_interest"
```

---

## Task 18: Package polish and exports

**Files:**
- Modify: `src/pocket_news/__init__.py`

- [ ] **Step 1: Write final `src/pocket_news/__init__.py`**

```python
"""pocket-news â€” fetch, deduplicate, and synthesize news with a local LLM."""
from __future__ import annotations
import logging

from .agent import NewsAgent
from .config import AgentConfig
from .exceptions import (
    ConfigurationError, FetcherError, OllamaUnavailableError,
    PocketNewsError, RateLimitError, SynthesisParseError,
)
from .models import Article, SourceCitation, SynthesizedArticle

__version__ = "0.1.0"

__all__ = [
    "NewsAgent", "AgentConfig",
    "PocketNewsError", "ConfigurationError", "FetcherError",
    "RateLimitError", "OllamaUnavailableError", "SynthesisParseError",
    "Article", "SourceCitation", "SynthesizedArticle",
    "configure_logging", "__version__",
]


def configure_logging(level: str = "INFO") -> None:
    """Configure pocket-news logging. Call once at application startup."""
    pkg = logging.getLogger("pocket_news")
    pkg.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not pkg.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        pkg.addHandler(h)
```

- [ ] **Step 2: Verify imports**

```bash
python -c "from pocket_news import NewsAgent, AgentConfig, SynthesizedArticle, PocketNewsError; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Run full test suite**

```bash
pytest tests/ -v -k "not integration"
```

Expected: all pass.

- [ ] **Step 4: Format**

```bash
ruff format src/ tests/ examples/
ruff check src/ tests/ examples/ --fix
```

- [ ] **Step 5: Run tests again post-format**

```bash
pytest tests/ -v -k "not integration"
```

Expected: still all pass.

- [ ] **Step 6: Commit**

```bash
git add src/pocket_news/__init__.py
git commit -m "feat: package exports and configure_logging() â€” v0.1.0 complete"
```

---

## Self-Review

- **Spec coverage**: All phases 0â€“11 mapped. `--setup` CLI in Task 16. `respx` used for all HTTP tests. Phase 12 deferred per design decision.
- **No placeholders**: All test and implementation code is complete and runnable.
- **Type consistency**: `Article`/`SourceCitation`/`SynthesizedArticle` defined Task 4, used consistently Tasks 5â€“18. `fetch_and_encode_image` â†’ `tuple[str|None, str|None]` matched in Task 15. `build_synthesis_prompt` â†’ `tuple[str,str]` matched in Task 14. `_parse_synthesis_output` â†’ `dict` with `headline`/`lead`/`body`/`key_points` matched in Task 15.
- **MediaStack quirk**: Task 7 checks `"error" in body` before assuming success.
- **Qwen3 `/no_think`**: Tasks 14 tests verify appended for qwen3, not for gemma, not when `disable_thinking_mode=False`.
- **`no_results` not cached**: `CacheStore.set` returns early, Task 12 tests verify, Task 15 skips cache write.

