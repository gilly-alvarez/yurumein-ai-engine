from pydantic import BaseModel, Field
from typing import List, Dict

class Learning_hub_Request(BaseModel):
    user_prompt: str


class VocabularyItem(BaseModel):
    word_phrase: str = Field(..., description="The word or phrase in the target language (e.g., Garifuna).")
    translation: str = Field(..., description="The translation in English or base language.")

PronunciationOptions = Dict[str, str]

class FocusWord(BaseModel):
    Question: str
    word_phrase: str
    option: PronunciationOptions =Field(..., description="The set of possible choices for the question.")
    answer: str =Field(...,description= "The correct answer. For multiple choice, this should be the key (e.g., 'A', 'B','C','D') corresponding to the correct option in QuizOptions ")


QuizOptions = Dict[str, str]

class QuizQuestion(BaseModel):
    id: int
    type: str
    question_text: str
    options: QuizOptions =Field(..., description="The set of possible choices for the question.")
    answer: str =Field(...,description= "The correct answer. For multiple choice, this should be the key (e.g., 'A', 'B','C','D') corresponding to the correct option in QuizOptions ")




class LessonUnit(BaseModel):
    text: str =Field(..., description="Give only text of the lesson nothing else")
    vocabulary_section:  List[VocabularyItem]
    pronunciation_section: List[FocusWord]
    quiz: List[QuizQuestion]


class LearningHub_Response(BaseModel):
    lesson_unit: LessonUnit
