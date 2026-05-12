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
