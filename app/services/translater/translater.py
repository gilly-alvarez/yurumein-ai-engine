import json
import os
from datetime import datetime
from typing import Dict, Optional, Set

from openai import OpenAI

from .translater_schema import llm_response
from app.core.config import settings
from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager
from app.services.lexicon.lexicon import YurumeinLexicon

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

_DECLINE_RESPONSE = {
    "en": (
        "A verified Garífuna translation is not available for this term. "
        "Try another word or phrase, or consult a Garífuna language expert."
    ),
    "es": (
        "No hay una traducción garífuna verificada disponible para este término. "
        "Intenta con otra palabra o frase, o consulta a un experto en el idioma garífuna."
    ),
}


class TranslationServiceUnavailableError(Exception):
    """Raised when the upstream AI provider is unavailable (quota, billing, rate-limit)."""
    pass


class Translater:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.vector_store = VectorStoreManager()
        self.lexicon = YurumeinLexicon()
        self._cached_garifuna_words: Optional[Set[str]] = None
        self._cache_loaded_at: Optional[datetime] = None

    def _load_garifuna_words(self) -> Set[str]:
        """Load all verified Garífuna word-forms from the lexicon for context grounding checks."""
        from sqlalchemy import text as sa_text
        try:
            with self.lexicon.engine.connect() as conn:
                rows = conn.execute(sa_text(
                    "SELECT garifuna FROM yurumein_lexicon WHERE garifuna IS NOT NULL AND garifuna != ''"
                )).mappings().all()
            words: Set[str] = set()
            for row in rows:
                for token in (row["garifuna"] or "").split():
                    w = token.strip(".,;:!?-").lower()
                    if len(w) > 2:
                        words.add(w)
            return words
        except Exception:
            return set()

    _CACHE_TTL_SECONDS = 600  # 10 minutes

    @property
    def _garifuna_words(self) -> Set[str]:
        now = datetime.utcnow()
        expired = (
            self._cache_loaded_at is None
            or (now - self._cache_loaded_at).total_seconds() > self._CACHE_TTL_SECONDS
        )
        if expired:
            self._cached_garifuna_words = self._load_garifuna_words()
            self._cache_loaded_at = now
        return self._cached_garifuna_words  # type: ignore[return-value]

    def _retrieve_context(self, text: str) -> str:
        """Hybrid retrieval (keyword + semantic) from the curated vector store."""
        try:
            seen: set[str] = set()
            chunks: list[str] = []

            for d in self.vector_store.keyword_search(query=text, k=5):
                chunk = (d.get("document") or "").strip()
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    chunks.append(chunk)

            for doc in self.vector_store.similarity_search(query=text, k=5):
                chunk = (doc.page_content or "").strip()
                if chunk and chunk not in seen:
                    seen.add(chunk)
                    chunks.append(chunk)

            return "\n---\n".join(chunks)
        except Exception:
            return ""

    def _context_has_garifuna_evidence(self, context: str) -> bool:
        """Returns True if the context contains at least one verified Garífuna word-form from the lexicon."""
        ctx_lower = context.lower()
        return any(word in ctx_lower for word in self._garifuna_words if len(word) > 2)

    def _output_contains_garifuna_forms(self, translation: str) -> bool:
        """
        Returns True only when at least one token of the translation is found in the
        lexicon's garifuna column word set. Anchors validation to labeled Garífuna data,
        not free-text context position — a Spanish word appearing as a gloss in a
        bilingual document cannot pass this check.
        """
        words = self._garifuna_words
        if not words:
            return False
        # Check the full translation and each comma/semicolon-separated segment
        segments = {translation.strip().lower()}
        for sep in (',', ';'):
            segments.update(seg.strip().lower() for seg in translation.split(sep))
        for segment in segments:
            for token in segment.split():
                w = token.strip(".,;:!?-").lower()
                if len(w) > 2 and w in words:
                    return True
        return False

    def get_system_prompt(self, context: str) -> str:
        return f"""You are **Yurumein AI**, a Garífuna language translator.

=== VERIFIED GARÍFUNA DICTIONARY CONTEXT ===
{context}
============================================

### Translation Rules
1. Use the verified dictionary context above as your PRIMARY source.
2. If a direct translation or equivalent is present in the context, use it exactly — do not paraphrase or approximate.
3. Do not guess, invent, or approximate Garífuna vocabulary or grammar. Garífuna is a living endangered language; inaccuracy causes cultural harm.
4. Maintain cultural fidelity and preserve traditional Garífuna forms.
5. Auto-detect input language when unspecified.
6. CRITICAL — when the target language is Garífuna: your output MUST be a Garífuna word or phrase found in the context above. If the context does not contain a Garífuna equivalent for the requested term, return {{"translation": ""}}. Never return a Spanish or English word as a Garífuna translation. No exceptions.

### Output format (strict JSON only)
{{"translation": "str"}}
No extra text, no markdown, no commentary outside the JSON object.
"""

    def get_user_prompt(self, base_language: str, target_language: str, text: str) -> str:
        return f"""Translate the following text from {base_language} to {target_language}:

Text:
\"{text}\"

Provide only the translated output in {target_language}."""

    def get_response(self, translation_info: Dict):
        try:
            text = translation_info["text"]
            base_language = translation_info["base_language"]
            target_language = translation_info["target_language"]
            ui_lang = (translation_info.get("ui_language") or "en").strip().lower()
            decline_msg = _DECLINE_RESPONSE.get(ui_lang, _DECLINE_RESPONSE["en"])

            is_garifuna_target = target_language.lower() == "garifuna"
            is_garifuna_source = base_language.lower() == "garifuna"

            # Lexicon-first exact lookup for Garífuna targets (EN/ES → Garífuna).
            if is_garifuna_target:
                r = self.lexicon.lookup(text, base_language)
                if r:
                    distinct_forms = []
                    for row in r:
                        form = (row.get("garifuna") or "").strip()
                        if form and form not in distinct_forms:
                            distinct_forms.append(form)
                    if distinct_forms:
                        return llm_response(translation="; ".join(distinct_forms))

            # Lexicon-first reverse lookup for Garífuna source (Garífuna → EN/ES).
            if is_garifuna_source and not is_garifuna_target:
                r = self.lexicon.reverse_lookup(text)
                if r:
                    col = "english" if "english" in target_language.lower() else "spanish"
                    forms = []
                    for row in r:
                        val = (row.get(col) or "").strip()
                        if val and val not in forms:
                            forms.append(val)
                    if forms:
                        return llm_response(translation="; ".join(forms))

            # Hybrid retrieval from the curated vector store.
            context = self._retrieve_context(text)

            # No context → verified database gap.
            if not context:
                return llm_response(translation=decline_msg)

            # For Garífuna target: context must contain at least one verified Garífuna word.
            # Spanish-only context cannot support a Garífuna translation.
            if is_garifuna_target and not self._context_has_garifuna_evidence(context):
                return llm_response(translation=decline_msg)

            messages = [
                {"role": "system", "content": self.get_system_prompt(context)},
                {
                    "role": "user",
                    "content": self.get_user_prompt(base_language, target_language, text),
                },
            ]

            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_output = completion.choices[0].message.content
            data = json.loads(raw_output)
            validated = llm_response(**data)

            translation_text = (validated.translation or "").strip()

            # Empty output means the model found no verified Garífuna equivalent in context
            # (following the system prompt instruction). Show the localized decline message —
            # not a 503 toast, which would imply the service is down.
            if not translation_text:
                return llm_response(translation=decline_msg)

            # For Garífuna target: validate that the output contains at least one word-form
            # found in the lexicon's garifuna column. A Spanish word appearing as a gloss
            # in the retrieved bilingual context cannot pass this check.
            if is_garifuna_target and not self._output_contains_garifuna_forms(translation_text):
                return llm_response(translation=decline_msg)

            return validated

        except TranslationServiceUnavailableError:
            raise
        except Exception as e:
            err = str(e)
            if "429" in err or "quota" in err.lower() or "credit" in err.lower():
                raise TranslationServiceUnavailableError(
                    "Translation service temporarily unavailable. Please try again later."
                )
            raise RuntimeError(f"Translation failed: {err}")
