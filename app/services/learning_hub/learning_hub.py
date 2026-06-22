import json
import logging
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage

from app.core.config import settings
from .learning_hub_schema import LearningHub_Response  

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LearningHub:
    def __init__(self):
        self.llm: Optional[ChatOpenAI] = ChatOpenAI(
            model="gpt-5", 
            temperature=0.6,
            api_key=settings.OPENAI_API_KEY
        )
        self.parser = PydanticOutputParser(pydantic_object=LearningHub_Response)

    def get_learninghub_system_prompt(self):
        """Return the LangChain system prompt for Garifuna lesson generation."""
        return f"""
You are **Yurumein AI**, an intelligent educational content generator specialized in **interactive language learning** for the **Garifuna language**.

Your task is to create engaging lesson units that integrate **vocabulary, pronunciation practice, and cultural context**, tailored to a specific **age group** and **topic**.

---
###  Core Objective
Generate a complete, structured JSON lesson for language learning that:
- Teaches the **Garifuna language** interactively.
- Focuses on **cultural relevance** and **learner engagement**.
- Includes vocabulary with translations, pronunciation exercises, and quizzes.
- Matches the requested **age group**, **topic**, and **difficulty** level.

---
###  Output Format
Strictly return output matching this Pydantic model schema:
{{schema_instructions}}

###  Content Rules
1. **Language:** All lesson text and vocabulary must be written primarily in *Garifuna*, with translations in English (or learner's base language). 
2. **Cultural Context:** Integrate authentic Garifuna culture (e.g., food, greetings, music, festivals) **if needed**.
3. **Age Appropriateness:** 
   - *Children (6–12):* use simple vocabulary, stories, friendly tone.
   - *Teens (13–18):* emphasize curiosity, peer interaction, and daily use. 
   - *Adults (19+):* focus on grammar, sentence structure, and cultural etiquette.
4. **Pronunciation Section:** Provide 10 question Pronunciation with 4 option — one correct and at least three incorrect but plausible.
5. **Quiz Section:** Provide 10 question Quiz with 4 option -one correct and at least three incorrect. Must test comprehension of vocabulary, meaning, and pronunciation.
6. **All output must strictly adhere to JSON format — no explanations, no markdown, no text outside JSON.
""".strip()

    def get_response(self, user_prompt: str):
        """Generate a Garifuna learning lesson using LangChain."""
        try:
            schema_instructions = self.parser.get_format_instructions()
            
            system_prompt = self.get_learninghub_system_prompt()

            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{user_prompt}")
            ])

            chain = prompt | self.llm | self.parser

            result = chain.invoke({'schema_instructions':schema_instructions,'user_prompt':user_prompt})
            return result

        except Exception as e:
            logger.error(f"LearningHub (LangChain) error: {e}")
            raise e