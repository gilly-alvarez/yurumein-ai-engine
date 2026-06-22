from pydantic import BaseModel


class Translater_text_Request(BaseModel):
    base_language: str
    target_language: str
    text: str

class Translater_text_Response(BaseModel):
    base_language: str
    target_language: str
    text: str
    translation: str

class llm_response(BaseModel):
    translation:str