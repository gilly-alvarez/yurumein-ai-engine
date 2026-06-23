import json
import os
from typing import Dict

from openai import OpenAI

from .translater_schema import llm_response
from app.core.config import settings
from app.services.utils.RAG_pipeline.vectore_db.pgvector_db import VectorStoreManager

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY


class Translater:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.vector_store = VectorStoreManager()

    def _retrieve_context(self, text: str) -> str:
        try:
            docs = self.vector_store.similarity_search(query=text, k=5)
            chunks = [doc.page_content for doc in docs if doc.page_content.strip()]
            return "\n---\n".join(chunks)
        except Exception:
            return ""

    def get_system_prompt(self, context: str) -> str:
        context_section = context if context else "No verified context found."
        return f"""You are **Yurumein AI**, a Garífuna language translator.

=== VERIFIED GARÍFUNA DICTIONARY CONTEXT ===
{context_section}
============================================

### Translation Rules
1. Use the verified dictionary context above as your PRIMARY source.
2. If a direct translation or equivalent is present in the context, use it exactly — do not paraphrase or approximate.
3. Do not guess, invent, or approximate Garífuna vocabulary or grammar. Garífuna is a living endangered language; inaccuracy causes cultural harm.
4. Maintain cultural fidelity and preserve traditional Garífuna forms.
5. Auto-detect input language when unspecified.

### When to decline
If the verified context above is missing, empty, or does not contain a clear basis for the requested translation, return this exact JSON and nothing else:
{{"translation": "This word or phrase is not currently available in the Yurumein AI verified Garífuna database. Please try another word or phrase, or consult a Garífuna language expert."}}

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
            context = self._retrieve_context(translation_info["text"])
            messages = [
                {"role": "system", "content": self.get_system_prompt(context)},
                {
                    "role": "user",
                    "content": self.get_user_prompt(
                        translation_info["base_language"],
                        translation_info["target_language"],
                        translation_info["text"],
                    ),
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
