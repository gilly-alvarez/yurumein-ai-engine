import re
import unicodedata
from typing import List, Dict
from sqlalchemy import create_engine, text
from app.core.config import settings

def normalize_term(value: str) -> str:
    value = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", value.strip().lower())

class YurumeinLexicon:
    def __init__(self):
        self.engine = create_engine(f"postgresql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")

    def ensure_table(self):
        sql = """CREATE TABLE IF NOT EXISTS yurumein_lexicon (id BIGSERIAL PRIMARY KEY, spanish TEXT, spanish_norm TEXT, garifuna TEXT NOT NULL, english TEXT, english_norm TEXT);"""

        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def lookup(self, term, language):
        self.ensure_table()
        norm = normalize_term(term)
        column = "spanish_norm" if language.lower() == "spanish" else "english_norm"
        sql = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sql, {"term": norm}).mappings().all()]
