from pydantic_settings import BaseSettings
from typing import Any


class Settings(BaseSettings):
    OPENAI_API_KEY: Any | None = None
    MODEL_NAME: Any
    COLLECTION_NAME : Any
    DB_USER:Any
    DB_PASS: Any
    DB_HOST:Any
    DB_PORT: Any
    DB_NAME:Any
    class Config:
        env_file = ".env"


settings = Settings()



