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
    # Required for the admin knowledge-gap endpoints.
    # Set to a strong random string in production. Requests without this key
    # receive 403 Forbidden. Leave empty only in local dev (endpoints return 503).
    ADMIN_API_KEY: str = ""
    class Config:
        env_file = ".env"


settings = Settings()



