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
        "1. EU AI Act enforcement launched\n2. Companies face August 2026 deadline\n• High-risk systems require full risk assessment",
    )
    result = _parse_synthesis_output(raw)
    assert result["key_points"][0] == "EU AI Act enforcement launched"
    assert result["key_points"][2] == "High-risk systems require full risk assessment"


def test_missing_section_raises():
    no_body = "===HEADLINE===\nHeadline\n\n===LEAD===\nLead.\n\n===KEY_POINTS===\n- Point\n"
    with pytest.raises(SynthesisParseError, match="body"):
        _parse_synthesis_output(no_body)
