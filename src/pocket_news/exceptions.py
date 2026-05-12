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
