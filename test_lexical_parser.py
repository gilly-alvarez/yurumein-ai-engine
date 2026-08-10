"""
Unit tests for the Phase 1 Chatbot Lexical Parser.

No DB, no LLM, no external dependencies — the parser module is pure stdlib.
conftest.py stubs are not needed for this file, but are harmless if loaded.

Covers:
  - All 6 supported patterns (3 English, 3 Spanish)
  - Case insensitivity
  - Whitespace tolerance
  - Terminal punctuation stripping
  - Accented Spanish terms preserved
  - Empty extracted term → matched=False
  - Unsupported lexical phrasing → matched=False
  - Unrelated cultural question → matched=False
"""

import pytest

from app.services.chatbot.lexical_parser import ParseResult, parse_lexical_request


# ---------------------------------------------------------------------------
# English patterns
# ---------------------------------------------------------------------------

class TestEnglishHowDoYouSay:
    def test_basic(self):
        r = parse_lexical_request("How do you say hello in Garífuna?")
        assert r.matched is True
        assert r.term == "hello"
        assert r.language == "english"
        assert r.pattern == "how_do_you_say"

    def test_lowercase_garifuna(self):
        r = parse_lexical_request("how do you say rain in garifuna?")
        assert r.matched is True
        assert r.term == "rain"
        assert r.language == "english"

    def test_multi_word_term(self):
        r = parse_lexical_request("How do you say good morning in Garífuna?")
        assert r.matched is True
        assert r.term == "good morning"

    def test_trailing_punctuation_stripped(self):
        r = parse_lexical_request("How do you say water in Garífuna.")
        assert r.matched is True
        assert r.term == "water"

    def test_no_trailing_question_mark(self):
        r = parse_lexical_request("How do you say fire in Garífuna")
        assert r.matched is True
        assert r.term == "fire"


class TestEnglishTranslateTo:
    def test_basic(self):
        r = parse_lexical_request("Translate mother to Garífuna.")
        assert r.matched is True
        assert r.term == "mother"
        assert r.language == "english"
        assert r.pattern == "translate_to"

    def test_lowercase(self):
        r = parse_lexical_request("translate fish to garifuna")
        assert r.matched is True
        assert r.term == "fish"

    def test_multi_word_term(self):
        r = parse_lexical_request("Translate the ocean to Garífuna.")
        assert r.matched is True
        assert r.term == "the ocean"

    def test_trailing_period_stripped(self):
        r = parse_lexical_request("Translate sun to Garífuna.")
        assert r.matched is True
        assert r.term == "sun"


class TestEnglishWordFor:
    def test_basic(self):
        r = parse_lexical_request("What is the Garífuna word for tree?")
        assert r.matched is True
        assert r.term == "tree"
        assert r.language == "english"
        assert r.pattern == "word_for"

    def test_lowercase_garifuna(self):
        r = parse_lexical_request("what is the garifuna word for moon?")
        assert r.matched is True
        assert r.term == "moon"

    def test_trailing_punctuation_stripped(self):
        r = parse_lexical_request("What is the Garífuna word for river.")
        assert r.matched is True
        assert r.term == "river"

    def test_multi_word_term(self):
        r = parse_lexical_request("What is the Garífuna word for the sea?")
        assert r.matched is True
        assert r.term == "the sea"


# ---------------------------------------------------------------------------
# Spanish patterns
# ---------------------------------------------------------------------------

class TestSpanishComoSeDice:
    def test_basic_with_opening_punctuation(self):
        r = parse_lexical_request("¿Cómo se dice agua en garífuna?")
        assert r.matched is True
        assert r.term == "agua"
        assert r.language == "spanish"
        assert r.pattern == "como_se_dice"

    def test_without_opening_punctuation(self):
        r = parse_lexical_request("Cómo se dice fuego en garífuna?")
        assert r.matched is True
        assert r.term == "fuego"
        assert r.language == "spanish"

    def test_without_accent_on_como(self):
        r = parse_lexical_request("¿Como se dice lluvia en garífuna?")
        assert r.matched is True
        assert r.term == "lluvia"

    def test_accented_term_preserved(self):
        r = parse_lexical_request("¿Cómo se dice árbol en garífuna?")
        assert r.matched is True
        assert r.term == "árbol"

    def test_multi_word_term(self):
        r = parse_lexical_request("¿Cómo se dice buenos días en garífuna?")
        assert r.matched is True
        assert r.term == "buenos días"


class TestSpanishTraduceAl:
    def test_basic(self):
        r = parse_lexical_request("Traduce madre al garífuna.")
        assert r.matched is True
        assert r.term == "madre"
        assert r.language == "spanish"
        assert r.pattern == "traduce_al"

    def test_lowercase(self):
        r = parse_lexical_request("traduce sol al garifuna")
        assert r.matched is True
        assert r.term == "sol"

    def test_accented_term_preserved(self):
        r = parse_lexical_request("Traduce corazón al garífuna.")
        assert r.matched is True
        assert r.term == "corazón"

    def test_trailing_period_stripped(self):
        r = parse_lexical_request("Traduce luna al garífuna.")
        assert r.matched is True
        assert r.term == "luna"


class TestSpanishPalabraPara:
    def test_basic_with_opening_punctuation(self):
        r = parse_lexical_request("¿Cuál es la palabra garífuna para mar?")
        assert r.matched is True
        assert r.term == "mar"
        assert r.language == "spanish"
        assert r.pattern == "palabra_para"

    def test_without_opening_punctuation(self):
        r = parse_lexical_request("Cuál es la palabra garífuna para río?")
        assert r.matched is True
        assert r.term == "río"

    def test_without_accent_on_cual(self):
        r = parse_lexical_request("¿Cual es la palabra garífuna para casa?")
        assert r.matched is True
        assert r.term == "casa"

    def test_accented_term_preserved(self):
        r = parse_lexical_request("¿Cuál es la palabra garífuna para canción?")
        assert r.matched is True
        assert r.term == "canción"

    def test_trailing_question_mark_stripped(self):
        r = parse_lexical_request("¿Cuál es la palabra garífuna para pez?")
        assert r.matched is True
        assert r.term == "pez"


# ---------------------------------------------------------------------------
# Rejection cases
# ---------------------------------------------------------------------------

class TestUnrelatedCulturalQuestion:
    def test_history_question(self):
        r = parse_lexical_request("What is the history of the Garífuna people?")
        assert r.matched is False
        assert r.term is None
        assert r.language is None
        assert r.pattern is None

    def test_cultural_tradition(self):
        r = parse_lexical_request("Tell me about Garífuna music traditions.")
        assert r.matched is False

    def test_spanish_cultural_question(self):
        r = parse_lexical_request("¿Cuáles son las tradiciones garífunas?")
        assert r.matched is False

    def test_greeting(self):
        r = parse_lexical_request("Hello, can you help me?")
        assert r.matched is False


class TestUnsupportedLexicalPhrasing:
    def test_wrong_english_phrasing(self):
        r = parse_lexical_request("What does hello mean in Garífuna?")
        assert r.matched is False

    def test_wrong_english_phrasing_2(self):
        r = parse_lexical_request("Give me the Garífuna translation of water.")
        assert r.matched is False

    def test_partial_pattern(self):
        r = parse_lexical_request("say hello in Garífuna")
        assert r.matched is False

    def test_wrong_spanish_phrasing(self):
        r = parse_lexical_request("¿Qué significa agua en garífuna?")
        assert r.matched is False

    def test_reverse_garifuna_lookup_not_supported(self):
        r = parse_lexical_request("How do you say Ünabu in English?")
        assert r.matched is False

    def test_reverse_garifuna_spanish_lookup_not_supported(self):
        r = parse_lexical_request("¿Qué significa Ünabu en español?")
        assert r.matched is False


class TestEmptyTerm:
    def test_empty_string(self):
        r = parse_lexical_request("")
        assert r.matched is False

    def test_whitespace_only(self):
        r = parse_lexical_request("   ")
        assert r.matched is False

    def test_pattern_with_no_term_content(self):
        # Term group captures only whitespace → rejected
        r = parse_lexical_request("How do you say   in Garífuna?")
        assert r.matched is False


class TestAccentPreservation:
    def test_spanish_acute_accent_in_term(self):
        r = parse_lexical_request("Traduce corazón al garífuna.")
        assert r.matched is True
        assert r.term == "corazón"

    def test_spanish_multiple_accents_in_term(self):
        r = parse_lexical_request("¿Cómo se dice también en garífuna?")
        assert r.matched is True
        assert r.term == "también"

    def test_spanish_tilde_in_term(self):
        r = parse_lexical_request("¿Cuál es la palabra garífuna para niño?")
        assert r.matched is True
        assert r.term == "niño"

    def test_english_term_without_accent_unaffected(self):
        r = parse_lexical_request("How do you say rain in Garífuna?")
        assert r.matched is True
        assert r.term == "rain"
