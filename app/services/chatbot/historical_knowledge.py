"""Citation and evidence handling for the separately governed Yurumein history draft.

This is NOT a lexicon authority. Uploaded historic FAQs are structured as
one complete question/answer per retrieval chunk, preserving the author-side
evidence state and original bibliographic reference identifiers.
"""
from __future__ import annotations

import re

_HISTORY_FILE_MARKER = "YURUMEIN_AI_HISTORIA_ES_EN_V1.2_AUDITADA"

# Display references are transcribed from the supplied audited history draft.
# These are bibliographic identifiers, not a claim of independent verification.
HISTORY_REFERENCES: dict[str, str] = {
    "S1": "Ivan Van Sertima, They Came Before Columbus (1976)",
    "S2": "UNESCO, Language, Dance and Music of the Garifuna (2001/2008)",
    "S3": "Smithsonian Folklife Festival, Garifuna: The Voice of the Ancestors",
    "S4": "Haslip-Viera, Ortiz de Montellano & Barbour, Current Anthropology 38(3), 1997",
    "S5": "The West India Committee, The Exile of the Garinagu / The Kalinago and the Garinagu",
    "S6": "al-Umari, Masalik al-absar (14th c.; direct source check pending)",
    "S7": "Endangered Language Alliance, Garifuna",
    "S8": "Alfonso Arrivillaga Cortes, Marcos Sanchez y la ocupacion garifuna de Labuga (2016)",
    "S9": "Belize Glessima Research, Garifuna History Timeline",
    "S10": "Belize Living Heritage, Garifuna Settlement Day & Yurumein",
    "S11": "Adrian Fraser, The Carib/Garifuna Story (2015)",
    "S12": "Fernandes et al., A genetic history of the pre-contact Caribbean, Nature (2021)",
    "S13": "Benn Torres et al., Analysis of biogeographic ancestry, AJPA (2019)",
    "S14": "Coe, Diehl & Stuiver, Dating of the San Lorenzo Phase, Science (1967)",
    "S15": "Adrian Fraser, Chatoyer, Joseph, Oxford (2016)",
}

_QA_HEADING = re.compile(
    r"(?m)^###\s+(yh001-(es|en)-q(\d{2}))\s+[—-]\s+(.+?)\s*$"
)
_FIELD = {
    "status": re.compile(r"(?m)^\*\*(?:Evidencia / Evidence|Evidence):\*\*\s*(.+?)\s*$"),
    "layer": re.compile(r"(?m)^\*\*(?:Capa / Layer|Layer):\*\*\s*([ABC])\s*$"),
    "sources": re.compile(r"(?m)^\*\*(?:Fuentes / Sources|Sources):\*\*\s*(.+?)\s*$"),
    "review_state": re.compile(r"(?m)^\*\*(?:Revisi[oó]n / Review|Review):\*\*\s*(.+?)\s*$"),
}


def is_history_file(path: str) -> bool:
    return _HISTORY_FILE_MARKER in (path or "").upper()


def source_ids(metadata: dict, content: str = "") -> list[str]:
    """Read explicit bibliographic IDs only; never infer a citation from similarity."""
    ids = metadata.get("sources") or []
    if isinstance(ids, str):
        ids = re.findall(r"S\d{1,2}", ids)
    if not ids:
        # Legacy plain-text chunks may contain only the final [S#] tokens.
        ids = re.findall(r"\[S(\d{1,2})\]", content)
        ids = [f"S{n}" for n in ids]
        # The full source list uses [S1] headings: do not treat it as a QA.
        if len(set(ids)) > 8:
            return []
    return list(dict.fromkeys(s for s in ids if s in HISTORY_REFERENCES))


def source_attribution(metadata: dict, content: str = "") -> str:
    return "; ".join(f"[{s}] {HISTORY_REFERENCES[s]}" for s in source_ids(metadata, content))


def layer_of(metadata: dict, content: str = "") -> str:
    if not is_history_file(metadata.get("source") or metadata.get("filename") or ""):
        return ""
    if metadata.get("layer") in ("A", "B", "C"):
        return metadata["layer"]
    match = _FIELD["layer"].search(content)
    return match.group(1) if match else ""


def parse_history_qa(text: str, source_path: str) -> list[tuple[str, dict]]:
    """Return complete, bilingual Q&A chunks with review/evidence metadata.

    Non-history files and files without our known explicit question IDs are
    left to the generic ingestion path. Text after a heading is bounded by
    the next Q&A or the next top-level section, so the bibliography is not
    accidentally bundled into the last answer.
    """
    if not is_history_file(source_path):
        return []

    headings = list(_QA_HEADING.finditer(text))
    chunks: list[tuple[str, dict]] = []

    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end():end]
        section = re.search(r"(?m)^(?:##\s+|---\s*$)", body)
        if section:
            body = body[:section.start()]

        fields: dict[str, str] = {}
        for key, matcher in _FIELD.items():
            found = matcher.search(body)
            fields[key] = found.group(1).strip() if found else ""

        # Fail closed on incomplete metadata. An unqualified historical claim
        # must not be indexed as a verified, published answer.
        if not fields["status"] or not fields["layer"] or not fields["sources"]:
            continue

        references = list(dict.fromkeys(re.findall(r"S\d{1,2}", fields["sources"])))
        references = [ref for ref in references if ref in HISTORY_REFERENCES]
        if not references:
            continue

        answer = body
        for matcher in _FIELD.values():
            answer = matcher.sub("", answer)
        answer = answer.strip()
        if not answer:
            continue

        doc_id, lang, number, question = heading.groups()
        meta = {
            "source": source_path,
            "filename": source_path.rsplit("/", 1)[-1],
            "doc_type": "historical_qa",
            "id": doc_id,
            "lang": lang,
            "question_number": int(number),
            "topic": "garifuna_history",
            "layer": fields["layer"],
            "status": fields["status"],
            "sources": references,
            "version": "1.2-audited-draft",
            "review_state": "pending_cultural_review",
            "publication_allowed": False,
            "lexicon_authority": False,
        }
        passage = (
            f"Question: {question.strip()}\n"
            f"Evidence: {fields['status']}\n"
            f"Answer: {answer}"
        )
        chunks.append((passage, meta))

    return chunks
