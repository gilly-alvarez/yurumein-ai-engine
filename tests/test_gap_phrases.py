"""
Tests for _GAP_PHRASES coverage.

Verifies that all four response patterns observed in production trigger
gap_acknowledged=True, and that a fully supported answer does not.

_GAP_PHRASES is defined inline here (identical to cahtbot_router.py) so
this test file runs without installing the full FastAPI dependency stack.
"""
import pytest

_GAP_PHRASES = (
    # English
    "do not have information", "don't have information",
    "do not provide information",
    "not available in", "not documented", "no documentation",
    "no information available", "unable to find", "cannot find",
    # Spanish — original
    "no encuentro información", "no tengo información",
    "no está documentado", "sin información",
    # Spanish — observed in production responses
    "no proporcionan información",
    "no proporciona información",
    "no documentan",
    "no hay información",
)


def _gap_acknowledged(response_text: str) -> bool:
    return any(p in response_text.lower() for p in _GAP_PHRASES)


# --- Production responses that MUST trigger detection ---

def test_english_do_not_provide_information():
    """Controlled test response — 1842 British census question."""
    response = (
        "The available sources do not provide information about the exact number of "
        "Garífuna speakers recorded in the 1842 British colonial census of St. Vincent. "
        "If you have any other questions about Garífuna language, history, or culture, "
        "feel free to ask!"
    )
    assert _gap_acknowledged(response), "Expected 'do not provide information' to match"


def test_spanish_no_proporcionan_informacion_q1():
    """Original Q1 — 1881 speakers in Livingston."""
    response = (
        "Los documentos disponibles no proporcionan información específica sobre cuántas "
        "personas hablaban garífuna en Livingston, Guatemala, en 1881. Sin embargo, se "
        "menciona que en tiempos más recientes, el idioma garífuna es hablado por "
        "aproximadamente 17,000 personas en Guatemala."
    )
    assert _gap_acknowledged(response), "Expected 'no proporcionan información' to match"


def test_spanish_no_proporcionan_informacion_q2():
    """Original Q2 — pronunciation differences Livingston vs Dangriga."""
    response = (
        "Los documentos disponibles no proporcionan información específica sobre las "
        "diferencias en la pronunciación del garífuna entre Livingston y Dangriga. "
        "Si bien se mencionan aspectos históricos y culturales de Livingston, no se "
        "aborda la variación lingüística entre estas dos localidades."
    )
    assert _gap_acknowledged(response), "Expected 'no proporcionan información' to match"


def test_spanish_no_documentan_q3():
    """Original Q3 — Dügü testimonies from Livingston elders."""
    response = (
        "Las fuentes disponibles no documentan testimonios específicos de ancianos "
        "garífunas de Livingston sobre los cambios en la ceremonia Dügü durante el "
        "siglo XIX. La información recuperada no proporciona detalles sobre este tema "
        "en particular."
    )
    assert _gap_acknowledged(response), "Expected 'no documentan' to match"


# --- A fully supported answer MUST NOT trigger detection ---

def test_fully_supported_answer_does_not_trigger():
    """Answer with solid evidence — should not register as a gap."""
    response = (
        "The Garífuna people (Garinagu) are descendants of the Island Caribs and "
        "West African peoples. They were exiled from St. Vincent (Yurumein) by the "
        "British in 1797 and settled on the Bay Islands of Honduras before spreading "
        "to Belize, Guatemala, and Nicaragua. Their language is part of the Arawakan "
        "family. This is documented in Libro Garífuna II, page 7."
    )
    assert not _gap_acknowledged(response), "Fully supported answer should not trigger gap detection"


# --- Edge cases for all phrases ---

def test_original_english_do_not_have_information():
    assert _gap_acknowledged("I do not have information about that specific record.")


def test_original_english_dont_have_information():
    assert _gap_acknowledged("I don't have information on that topic.")


def test_no_documentation():
    assert _gap_acknowledged("There is no documentation available for this claim.")


def test_sin_informacion():
    assert _gap_acknowledged("Sin información disponible en las fuentes consultadas.")


def test_no_hay_informacion():
    assert _gap_acknowledged("No hay información en los documentos sobre ese tema específico.")


def test_no_proporciona_informacion_singular():
    assert _gap_acknowledged(
        "El texto no proporciona información sobre la fecha exacta del evento."
    )


def test_no_esta_documentado():
    assert _gap_acknowledged("Este hecho no está documentado en las fuentes disponibles.")
