# pocket-news Design Spec

**Date:** 2026-05-11
**Status:** Approved

## Overview

`pocket-news` is a pip-installable Python library (`pip install pocket-news`, `from pocket_news import NewsAgent`) that fetches news articles about a topic from three news APIs in parallel, deduplicates results, and uses a local Ollama model to synthesize them into a single coherent article. Supports optional translation to any supported language, base64 hero image embedding, and source citations.

Designed as a building block: give it one topic, get one typed Pydantic article back. No UI, no opinions on persistence.

## Source Spec Files

Full contracts are defined in `.claude/docs/`:
- `README (2).md` — public API contract and usage examples
- `ARCHITECTURE (1).md` — module layout, design decisions, data flow
- `API_REFERENCE.md` — World News API, The News API, MediaStack request/response specs + unified `Article` model
- `SYNTHESIZER.md` — Ollama integration, prompt template, `/no_think` directive, output parsing
- `IMPLEMENTATION_PLAN.md` — 12-phase build order with per-phase validation steps

## Name

| Context | Value |
|---|---|
| PyPI package | `pocket-news` |
| Python import | `pocket_news` |
| Module directory | `src/pocket_news/` |
| Public class | `NewsAgent` |

All references to `newsweave` in the source spec files are superseded by `pocket_news` / `pocket-news`.

## Tech Stack

- Python 3.9+
- `httpx` — HTTP client (sync; parallel fetch via `ThreadPoolExecutor`)
- `pydantic` v2 — all structured data models
- `ollama` — official Python client for Ollama
- `python-dateutil` — cross-source date parsing
- `platformdirs` — local cache directory
- `respx` — HTTP mocking in tests (dev dependency)
- Standard library `logging`, `concurrent.futures`, `difflib`, `hashlib`, `re`

No other runtime dependencies.

## Module Layout

```
src/pocket_news/
├── __init__.py              # Public exports + __version__
├── agent.py                 # NewsAgent orchestrator
├── cache.py                 # CacheStore (local file cache)
├── config.py                # AgentConfig (Pydantic v2)
├── dedupe.py                # ArticleDeduplicator
├── exceptions.py            # Error hierarchy
├── images.py                # fetch_and_encode_image()
├── languages.py             # Language name → ISO 639-1
├── models.py                # Article, SourceCitation, SynthesizedArticle
├── prompts.py               # Prompt templates + build_synthesis_prompt()
├── synthesizer.py           # OllamaSynthesizer
├── cli.py                   # argparse CLI
├── __main__.py              # → cli.main()
├── _secrets.example.py      # Checked in template
└── fetchers/
    ├── __init__.py          # FETCHER_REGISTRY
    ├── base.py              # BaseFetcher + DOMAIN_TO_OUTLET
    ├── world_news.py
    ├── the_news.py
    └── mediastack.py
```

## Data Flow

```
research(topic, language, length)
  │
  ├─ normalize_language() → (iso_code, display_name)
  ├─ map length → word_count
  ├─ CacheStore.get() → hit? return immediately
  │
  ├─ [ThreadPoolExecutor] fetch all configured fetchers concurrently
  │     each fetcher: FetcherError/RateLimitError caught → log + "failed: <reason>"
  │     all empty/failed → SynthesizedArticle(status="no_results"), no cache write, return
  │
  ├─ ArticleDeduplicator.deduplicate()
  │     URL canonicalization → exact title → 0.85 similarity (difflib)
  │
  ├─ select featured_image_url (first article with non-empty image_url)
  ├─ fetch_and_encode_image() if config.fetch_image_data (never raises)
  │
  ├─ OllamaSynthesizer.synthesize()
  │     builds prompt → appends /no_think if qwen3 → ollama.Client.chat()
  │     strips <think> blocks → parses ===SECTION=== delimiters
  │     SynthesisParseError if required sections missing
  │
  ├─ build SynthesizedArticle (status="ok" or "partial")
  ├─ CacheStore.set()
  └─ return
```

## Error Handling

Two exceptions propagate to the caller:

| Exception | When |
|---|---|
| `OllamaUnavailableError` | Ollama not running, wrong host, or model not pulled |
| `SynthesisParseError` | Model returned output that couldn't be parsed |

Both subclass `PocketNewsError` (the renamed base exception — replaces `NewsweaveError` from the original spec everywhere).

Soft failures — one fetcher down, image fetch timeout, rate limit — are absorbed internally. Reflected in `fetcher_status` dict. Result gets `status="partial"` if synthesis ran, `status="no_results"` only if every fetcher returned nothing.

## Testing Strategy

- **Unit tests** (`tests/`): every module in isolation, `respx` mocks all HTTP. Covers models, config resolution, language normalization, dedup edge cases, image error paths, cache TTL/round-trip, prompt building, output parser.
- **Integration tests**: gated with `pytest.mark.skipif` on API key env vars. One real HTTP call, two articles. Only run when keys are present.
- **Agent tests**: mock fetchers + mock synthesizer + mock image fetcher — verify full pipeline state machine.
- **No Ollama in CI**: synthesizer unit tests mock `ollama.Client`. Real Ollama calls are manual-only.

## Build Order

Phases 0–11 from `IMPLEMENTATION_PLAN.md`, in order. Phase 12 (CI, wheel, CHANGELOG) deferred.

Each phase completes its validation gate before the next phase begins.

## Key Decisions

- **Sync public API, parallel internals**: `ThreadPoolExecutor` for network I/O; `research_batch()` adds topic-level concurrency.
- **`status` field, not raise, for no-results**: keeps the parent project's topic loop simple.
- **Single LLM call for synthesis + translation**: modern multilingual models handle read-English/write-target natively; two calls would double latency.
- **Base64 images**: parent project can render without a second HTTP call that might 404 or be hotlink-blocked.
- **`_secrets.py` pattern**: no extra dependency, editor autocomplete works, env vars and kwargs still override.
- **`--setup` CLI command**: `python -m pocket_news --setup` prompts for each API key interactively and writes them to `src/pocket_news/_secrets.py`. Guided alternative to editing the file manually. Resolution chain unchanged.
- **Default model `qwen3:14b`**: best multilingual + structured output in 14B class; `/no_think` directive suppresses internal monologue for clean structured output.
