import json
from openai import OpenAI
from .translater_schema import llm_response
from app.core.config import settings
from typing import Dict, Optional
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY

class Translater:
    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )
    
    def get_system_prompt(self):
        return """
You are **Yurumein AI**, a professional multilingual translator specializing in Garifuna, Spanish, and English.

### Core Responsibilities
1. Translate text accurately between Garifuna, Spanish, and English.
2. Maintain cultural fidelity.
3. Preserve traditional Garifuna forms.
4. Ensure fluency in English/Spanish.
5. Return only translated text.

### Rules
- Auto-detect input language when unspecified.
- Avoid literal translation if meaning changes.
- If no equivalent exists, add a short explanation in parentheses.

### Output format (strict JSON)
{
"translation": "str(translated text only)"
}
No extra text, no markdown, no commentary.
"""


    def get_user_prompt(self, base_language: str, target_language: str, text: str):
        return f"""
Translate the following text from {base_language} to {target_language}:

Text:
\"{text}\"

Provide only the translated output in {target_language}.
"""

    def get_response(self, translation_info: Dict):
        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {
                    "role": "user",
                    "content": self.get_user_prompt(
                        translation_info["base_language"],
                        translation_info["target_language"],
                        translation_info["text"]
                    ),
                },
            ]
            import time

            start = time.time()
            print("LLM time:", time.time() - start)

            completion = self.client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw_output = completion.choices[0].message.content
            data = json.loads(raw_output)
            print("data",data)
            # Validate with your Pydantic schema
            validated = llm_response(**data)

            return validated

        except Exception as e:
            raise RuntimeError(f"Translation failed: {str(e)}")



