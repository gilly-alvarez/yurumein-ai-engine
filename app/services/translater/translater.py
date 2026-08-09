import json
import os
from typing import Dict

from openai import OpenAI

from .translater_schema import llm_response
from app.core.config import settings
from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager
from app.services.lexicon.lexicon import YurumeinLexicon

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

# Returned when the curated vector store has no grounded context for the
# requested term or phrase. The LLM is never called in this path.
_DECLINE_RESPONSE = (
    "This word or phrase is not currently available in the Yurumein AI "
    "verified Garífuna database. Please try another word or phrase, or "
    "consult a Garífuna language expert."
)


class Translater:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.vector_store = VectorStoreManager()
        # NOTE: Lexicon entries represent the Enríquez documented source dataset.
        # Row-level independent verification is not yet in the schema and will
        # be addressed in a later controlled schema migration.
        self.lexicon = YurumeinLexicon()

    def _retrieve_context(self, text: str) -> str:
        """Keyword retrieval from the curated vector store. Returns empty string on miss or error."""
        try:
            docs = self.vector_store.keyword_search(query=text, k=5)
            chunks = [d["document"] for d in docs if (d.get("document") or "").strip()]
            return "\n---\n".join(chunks)
        except Exception:
            return ""

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

            # Lexicon-first exact lookup for Garífuna targets.
            # Authoritative relative to the imported source dataset; independent
            # verification governance is deferred to a later schema migration.
            if target_language.lower() == "garifuna":
                r = self.lexicon.lookup(text, base_language)
                if r:
                    return llm_response(translation=r[0]["garifuna"])

            # Keyword retrieval from the curated vector store.
            context = self._retrieve_context(text)

            # Decline without LLM inference when no grounded context exists.
            if not context:
                return llm_response(translation=_DECLINE_RESPONSE)

            messages = [
                {"role": "system", "content": self.get_system_prompt(context)},
                {
                    "role": "user",
                    "content": self.get_user_prompt(base_language, target_language, text),
                },
            ]

            completion = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_output = completion.choices[0].message.content
            data = json.loads(raw_output)
            validated = llm_response(**data)
            return validated

        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")
