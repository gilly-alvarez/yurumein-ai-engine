import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.chatbot.historical_knowledge import (
    HISTORY_REFERENCES, is_history_file, layer_of, parse_history_qa,
    source_attribution, source_ids,
)

NAME = "Yurumein_AI_Historia_ES_EN_v1.2_Auditada_PARA_SUBIR.txt"
SAMPLE = """
### yh001-es-q09 — ¿Qué sostiene Van Sertima?

**Evidencia / Evidence:** HIPÓTESIS DEL AUTOR
**Capa / Layer:** B
**Fuentes / Sources:** [S1], [S4]
**Revisión / Review:** pending_cultural_review

Van Sertima plantea viajes. [S1, S4]

### yh001-en-q09 — What does Van Sertima claim?

**Evidencia / Evidence:** AUTHOR'S HYPOTHESIS
**Capa / Layer:** B
**Fuentes / Sources:** [S1], [S4]
**Revisión / Review:** pending_cultural_review

Van Sertima proposed voyages. [S1, S4]

## FUENTES / SOURCES
[S1] Van Sertima
"""


def test_history_source_detection():
    assert is_history_file(NAME)
    assert not is_history_file("DICCIONARIO GARIFUNA.pdf")


def test_history_qa_preserves_complete_references():
    chunks = parse_history_qa(SAMPLE, NAME)
    assert len(chunks) == 2
    first, meta = chunks[0]
    assert "Van Sertima plantea viajes" in first
    assert meta["id"] == "yh001-es-q09"
    assert meta["lang"] == "es"
    assert meta["layer"] == "B"
    assert meta["sources"] == ["S1", "S4"]
    assert not meta["publication_allowed"]
    assert not meta["lexicon_authority"]
    assert meta["review_state"] == "pending_cultural_review"
    assert "FUENTES / SOURCES" not in chunks[-1][0]


def test_history_attribution_is_original_not_filename():
    content, meta = parse_history_qa(SAMPLE, NAME)[0]
    sources = source_attribution(meta, content)
    assert "Ivan Van Sertima" in sources
    assert "Haslip-Viera" in sources
    assert NAME not in sources
    assert layer_of(meta, content) == "B"


def test_no_fabricated_history_references():
    assert source_ids({"sources": ["S1", "S999"]}) == ["S1"]
    assert source_attribution({}, "no evidence ID") == ""
    assert len(HISTORY_REFERENCES) == 15


def test_non_history_file_not_parsed_as_qa():
    assert parse_history_qa(SAMPLE, "other.txt") == []
