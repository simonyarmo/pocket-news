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
