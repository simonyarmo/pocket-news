# tests/test_images.py
import base64
import httpx
import pytest
import respx
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
