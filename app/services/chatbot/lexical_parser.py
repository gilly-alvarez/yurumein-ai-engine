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
    """Strip whitespace, surrounding quotes, and trailing sentence punctuation."""
    term = raw.strip()
    # Remove surrounding single or double quotes that the user typed
    if len(term) >= 2 and term[0] in ("'", '"') and term[-1] == term[0]:
        term = term[1:-1].strip()
    term = _TERMINAL_PUNCT.sub("", term)
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


@dataclass
class PageRequest:
    matched: bool
    source_pattern: str | None
    display_name: str | None
    pages: tuple


# Book aliases → (source ILIKE pattern, display name).
# Aliases are lowercase; matching uses text.lower().
_BOOK_MAP = [
    (["libro serie i", "serie i"], "%LIBRO SERIE I%", "Libro Serie I"),
    (
        ["libro garifuna ii", "libro garífuna ii", "garífuna ii", "garifuna ii"],
        "%GARIFUNA II%",
        "Libro Garífuna II",
    ),
]

_PAGE_TRIGGER_RE = re.compile(
    r"\b(?:page|pages|p[áa]gina|p[áa]ginas)\b", re.IGNORECASE
)
_PAGE_NUMBER_RE = re.compile(r"\d+")


def parse_page_request(text: str) -> PageRequest:
    """
    Parse a request naming a recognized book plus explicit page number(s).

    Triggers only when BOTH a recognized book alias AND a page-trigger word
    plus at least one integer are present. Returns matched=True with
    source_pattern, display_name, and a pages tuple of ints.
    """
    if not text or not text.strip():
        return PageRequest(matched=False, source_pattern=None, display_name=None, pages=())

    text_lower = text.lower()

    if not _PAGE_TRIGGER_RE.search(text):
        return PageRequest(matched=False, source_pattern=None, display_name=None, pages=())

    page_numbers = _PAGE_NUMBER_RE.findall(text)
    if not page_numbers:
        return PageRequest(matched=False, source_pattern=None, display_name=None, pages=())

    for aliases, source_pattern, display_name in _BOOK_MAP:
        if any(alias in text_lower for alias in aliases):
            return PageRequest(
                matched=True,
                source_pattern=source_pattern,
                display_name=display_name,
                pages=tuple(int(p) for p in page_numbers),
            )

    return PageRequest(matched=False, source_pattern=None, display_name=None, pages=())
