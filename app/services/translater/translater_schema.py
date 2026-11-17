from typing import Optional
from pydantic import BaseModel


class Translater_text_Request(BaseModel):
    base_language: str
    target_language: str
    text: str
    ui_language: Optional[str] = "en"


class Translater_text_Response(BaseModel):
    base_language: str
    target_language: str
    text: str
    translation: str
    # match_type indicates which tier produced the translation.
    # Additive field — clients that read only 'translation' are unaffected.
    #
    # Forward lookup (EN/ES → GAR):
    #   'exact'  — Tier 1: SQL equality on indexed english_norm / spanish_norm column
    #   'folded' — Tier 2: LIKE annotation-stripping fallback (e.g. 'handsome' matched
    #              'handsome (masculine)' stored entry via prefix)
    #
    # Reverse lookup (GAR → EN/ES):
    #   'exact'  — Tier 1: SQL equality on indexed garifuna_norm column (NFC lowercase)
    #   'folded' — Tier 2: in-memory diacritic-stripped scan; unaccented input matched
    #              an accented lexicon entry (e.g. 'Banguli' → 'Bángüli')
    #
    # LLM / no-match paths:
    #   'llm'    — LLM-generated from RAG context; no lexicon match was found
    #   'none'   — no match and no grounded LLM result; response is a decline message
    match_type: Optional[str] = None


class llm_response(BaseModel):
    translation: str
    match_type: Optional[str] = None
