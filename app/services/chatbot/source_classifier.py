"""
Maps document source paths to governance roles and display names.

Each source has:
  role       — the classification used for retrieval routing
  display    — human-readable name for citations
  priority   — lower = higher priority (0 = structured lexicon)
  attribution — author/origin string for attribution statements
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SourceMeta:
    role: str
    display: str
    priority: int
    attribution: str


# Ordered: first match wins, so put the most specific keys first.
_ROLE_MAP: list[tuple[str, SourceMeta]] = [
    (
        "LIBRO SERIE I",
        SourceMeta(
            role="FOUNDATIONAL_LANGUAGE_EDUCATION",
            display="Libro Serie I",
            priority=2,
            attribution="Mauricio Enríquez",
        ),
    ),
    (
        "LIBRO GARIFUNA II FORMATO DIGITAL",
        SourceMeta(
            role="ADVANCED_LANGUAGE_AND_CULTURAL_CONTEXT",
            display="Libro Garífuna II",
            priority=3,
            attribution="Elmer Mauricio Enríquez Bermúdez",
        ),
    ),
    (
        "DICCIONARIO GARIFUNA",
        SourceMeta(
            role="DICTIONARY",
            display="Diccionario Garífuna",
            priority=1,
            attribution="",
        ),
    ),
    (
        "YURUMEIN LEXICON",
        SourceMeta(
            role="STRUCTURED_LEXICON",
            display="Yurumein Lexicon",
            priority=0,
            attribution="",
        ),
    ),
    (
        "GARIFUNA LANGUAGE OVERVIEW",
        SourceMeta(
            role="SUPPLEMENTARY",
            display="Garífuna Language Overview",
            priority=4,
            attribution="",
        ),
    ),
]


def classify_source(source_path: str) -> Optional[SourceMeta]:
    """Return the SourceMeta for a source path, or None if unrecognized."""
    if not source_path:
        return None
    upper = source_path.upper()
    for key, meta in _ROLE_MAP:
        if key in upper:
            return meta
    return None


def source_display_name(metadata: dict) -> str:
    """Return a citation-ready display string from chunk metadata."""
    source_path = metadata.get("filename") or metadata.get("source") or ""
    page_label = metadata.get("page_label")
    page_raw = metadata.get("page")

    meta = classify_source(source_path)
    display = meta.display if meta else source_path.rsplit("/", 1)[-1]

    if page_label is not None:
        return f"{display} — page {page_label}"
    if page_raw is not None:
        return f"{display} — page {int(page_raw) + 1}"
    return display
