"""OllamaSynthesizer — wraps the Ollama client for news synthesis."""
from __future__ import annotations
import logging
import re
import httpx
from ollama import Client
from .exceptions import OllamaUnavailableError, SynthesisParseError
from .models import Article
from .prompts import build_synthesis_prompt

logger = logging.getLogger(__name__)

SECTION_PATTERN = re.compile(r"===\s*([A-Z_]+)\s*===")
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)
_BULLET_PREFIX = re.compile(r"^[\-\*•]|\d+[\.\)]\s*")


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
