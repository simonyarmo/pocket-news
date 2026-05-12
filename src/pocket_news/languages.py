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
