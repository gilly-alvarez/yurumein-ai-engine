"""
Detects the intent of a user query and returns the ordered list of
preferred source roles to use for retrieval boosting.
"""
from __future__ import annotations

import re
from enum import Enum


class Intent(str, Enum):
    ANTHEM                = "ANTHEM"
    TRANSLATE             = "TRANSLATE"
    DEFINE                = "DEFINE"
    PRONUNCIATION         = "PRONUNCIATION"
    FOUNDATIONAL_LANG     = "FOUNDATIONAL_LANG"
    ADVANCED_GRAMMAR      = "ADVANCED_GRAMMAR"
    SENTENCE_CONSTRUCTION = "SENTENCE_CONSTRUCTION"
    NUMBERING             = "NUMBERING"
    CULTURAL_HISTORICAL   = "CULTURAL_HISTORICAL"
    GENERAL               = "GENERAL"


# Preferred source roles in priority order per intent.
# Empty list means no preference — all sources compete equally.
_INTENT_SOURCES: dict[Intent, list[str]] = {
    Intent.ANTHEM:                ["NATIONAL_ANTHEM"],
    Intent.TRANSLATE:             ["STRUCTURED_LEXICON", "DICTIONARY"],
    Intent.DEFINE:                ["STRUCTURED_LEXICON", "DICTIONARY", "FOUNDATIONAL_LANGUAGE_EDUCATION"],
    Intent.PRONUNCIATION:         ["FOUNDATIONAL_LANGUAGE_EDUCATION", "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"],
    Intent.FOUNDATIONAL_LANG:     ["FOUNDATIONAL_LANGUAGE_EDUCATION"],
    Intent.ADVANCED_GRAMMAR:      ["ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT", "FOUNDATIONAL_LANGUAGE_EDUCATION"],
    Intent.SENTENCE_CONSTRUCTION: ["ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"],
    Intent.NUMBERING:             ["FOUNDATIONAL_LANGUAGE_EDUCATION", "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT"],
    Intent.CULTURAL_HISTORICAL:   ["GARIFUNA_HISTORY_AND_IDENTITY", "ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT", "COMMUNITY_ARCHIVE"],
    Intent.GENERAL:               [],
}

# Patterns ordered most-specific to least-specific.
# First match wins.
_INTENT_PATTERNS: list[tuple[re.Pattern, Intent]] = [
    # National anthem — most specific; must precede cultural/historical
    (re.compile(
        r'\b(national\s+anthem|himno\s+nacional|hymn|himno\s+gar[ií]funa'
        r'|gar[ií]funa\s+anthem|himno\s+de\s+guatemala|guatemala\s+anthem'
        r'|anthem\s+of\s+guatemala|letra\s+del\s+himno|wadimalu)\b',
        re.I,
    ), Intent.ANTHEM),

    # Counting / numbers
    (re.compile(
        r'\b(count(?:ing)?|numbers?|n[uú]meros?|contar|numeraci[oó]n|digits?|arithmetic|math(?:ematics)?)\b',
        re.I,
    ), Intent.NUMBERING),

    # Pronunciation / phonetics / alphabet
    (re.compile(
        r'\b(pronunci|pronunciation|vowels?|consonants?|alphabet|sounds?|phonet|fonética|fonolog'
        r'|letras?|vocales?|consonantes?|syllables?|s[ií]labas?|stress|accent)\b',
        re.I,
    ), Intent.PRONUNCIATION),

    # Sentence structure / syntax
    (re.compile(
        r'\b(sentence(?:\s+construction|\s+structure)?|oraci[oó]n|construct|syntax|sintaxis'
        r'|armado?\s+de|building\s+sentences?|how\s+(?:do|does|are)\s+sentences?)\b',
        re.I,
    ), Intent.SENTENCE_CONSTRUCTION),

    # Advanced grammar / morphology
    (re.compile(
        r'\b(gramm?ar|gramática|morpholog|advanced|avanzado|complex|complejo|suffix|prefix)\b',
        re.I,
    ), Intent.ADVANCED_GRAMMAR),

    # Cultural / historical
    (re.compile(
        r'\b(histor(?:y|ia)?|origin(?:es|s)?|culture|cultura|tradici[oó]n|tradition'
        r'|heritage|patrimonio|people|pueblo|migration|migraci[oó]n|ancestor|antepasado'
        r'|garifuna\s+people|diaspora|deportation|deportaci[oó]n'
        r'|africa[n]?|pre-?columb(?:ian)?|before\s+columbus|contact\s+theor'
        r'|shipwreck|resist(?:ance|encia)?|colonial(?:ism|ismo)?)\b',
        re.I,
    ), Intent.CULTURAL_HISTORICAL),

    # Definition / meaning
    (re.compile(
        r'\b(what\s+does\s+\S+\s+mean|what\s+is\s+the\s+meaning|significa|define|definici[oó]n'
        r'|meaning|significado|qu[eé]\s+significa)\b',
        re.I,
    ), Intent.DEFINE),

    # Foundational / teaching / basic concepts
    (re.compile(
        r'\b(teach|ense[ñn]|learn|aprender|basic|b[aá]sico|fundament|introduc'
        r'|lesson|lecci[oó]n|beginner|principiante|how\s+(?:do|does)\s+.*work)\b',
        re.I,
    ), Intent.FOUNDATIONAL_LANG),
]


def detect_intent(query: str) -> Intent:
    """
    Classify query intent using deterministic pattern matching.
    Returns the first matching intent, or GENERAL if none match.
    """
    if not query:
        return Intent.GENERAL
    for pattern, intent in _INTENT_PATTERNS:
        if pattern.search(query):
            return intent
    return Intent.GENERAL


def preferred_roles_for(intent: Intent) -> list[str]:
    """Return the ordered list of preferred source roles for an intent."""
    return list(_INTENT_SOURCES.get(intent, []))
