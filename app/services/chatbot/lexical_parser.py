"""
Deterministic English/Spanish → Garífuna lexical-request parser.
Also detects definition requests (Garífuna → EN/ES reverse lookup).

Phase 1 only. No LLM, no DB, no external dependencies.

Returns a ParseResult indicating whether the input matches a supported
lexical-lookup pattern, and if so, what term was requested and in which
source language.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class ParseResult:
    matched: bool
    term: Optional[str]
    language: Optional[str]   # "english" | "spanish" | "garifuna" | None
    pattern: Optional[str]


# Strips trailing sentence-ending punctuation from extracted terms.
_TERMINAL_PUNCT = re.compile(r'[.?!;,]+$')

# Each entry: (compiled regex, language, pattern_name)
# Groups with .+? (non-greedy) capture terms that appear mid-sentence.
# Groups with .+ (greedy) capture terms that appear at end-of-sentence and
# rely on _clean_term to strip trailing punctuation.
_TRANSLATE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (
        re.compile(
            r'how\s+do\s+you\s+say\s+(.+?)\s+in\s+gar[ií]funa\b',
            re.IGNORECASE,
        ),
        "english",
        "how_do_you_say",
    ),
    (
        re.compile(
            r'translate\s+(.+?)\s+to\s+gar[ií]funa\b',
            re.IGNORECASE,
        ),
        "english",
        "translate_to",
    ),
    (
        re.compile(
            r'what\s+is\s+the\s+gar[ií]funa\s+word\s+for\s+(.+)',
            re.IGNORECASE,
        ),
        "english",
        "word_for",
    ),
    (
        re.compile(
            r'¿?\s*c[oó]mo\s+se\s+dice\s+(.+?)\s+en\s+gar[ií]funa\b',
            re.IGNORECASE,
        ),
        "spanish",
        "como_se_dice",
    ),
    (
        re.compile(
            r'traduce\s+(.+?)\s+al\s+gar[ií]funa\b',
            re.IGNORECASE,
        ),
        "spanish",
        "traduce_al",
    ),
    (
        re.compile(
            r'¿?\s*cu[aá]l\s+es\s+la\s+palabra\s+gar[ií]funa\s+para\s+(.+)',
            re.IGNORECASE,
        ),
        "spanish",
        "palabra_para",
    ),
]

# Definition patterns: "what does X mean", "define X", "¿qué significa X?"
# These trigger a Garífuna reverse lookup (term → EN/ES).
_DEFINE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r'what\s+does\s+(.+?)\s+mean', re.IGNORECASE),
        "define_what_does",
    ),
    (
        re.compile(r'^define\s+(.+)', re.IGNORECASE),
        "define_explicit",
    ),
    (
        re.compile(r'¿?\s*qu[eé]\s+significa\s+(.+)', re.IGNORECASE),
        "define_que_significa",
    ),
    (
        re.compile(r'¿?\s*qu[eé]\s+quiere\s+decir\s+(.+)', re.IGNORECASE),
        "define_que_quiere_decir",
    ),
]

_NO_MATCH = ParseResult(matched=False, term=None, language=None, pattern=None)


def _clean_term(raw: str) -> Optional[str]:
    """Strip surrounding whitespace and trailing sentence punctuation."""
    term = _TERMINAL_PUNCT.sub("", raw.strip())
    return term if term else None


def parse_lexical_request(text: str) -> ParseResult:
    """
    Parse a natural-language EN/ES → Garífuna translation request.

    Returns a ParseResult with matched=True only when the input matches one
    of the six supported patterns and the extracted term is non-empty.
    No DB access, no LLM, fully deterministic.
    """
    if not text or not text.strip():
        return _NO_MATCH

    for regex, language, pattern_name in _TRANSLATE_PATTERNS:
        m = regex.search(text)
        if m:
            term = _clean_term(m.group(1))
            if not term:
                return _NO_MATCH
            return ParseResult(
                matched=True,
                term=term,
                language=language,
                pattern=pattern_name,
            )

    return _NO_MATCH


def parse_definition_request(text: str) -> ParseResult:
    """
    Parse a Garífuna definition request.

    Matches "what does X mean", "define X", "¿qué significa X?" and similar.
    The extracted term is treated as a Garífuna word for reverse lookup.
    Returns matched=True only when the extracted term is non-empty.
    """
    if not text or not text.strip():
        return _NO_MATCH

    for regex, pattern_name in _DEFINE_PATTERNS:
        m = regex.search(text)
        if m:
            term = _clean_term(m.group(1))
            if not term:
                return _NO_MATCH
            return ParseResult(
                matched=True,
                term=term,
                language="garifuna",
                pattern=pattern_name,
            )

    return _NO_MATCH
