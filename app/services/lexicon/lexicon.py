import re
import unicodedata
from sqlalchemy import create_engine, text
from app.core.config import settings

_SUPPORTED_LANGUAGES = frozenset({"english", "spanish"})


def normalize_term(value: str) -> str:
    # NFD decomposes accented characters into base + combining marks;
    # filtering category Mn strips the combining marks (accent-folding).
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.strip().lower())

class YurumeinLexicon:
    def __init__(self):
        self.engine = create_engine(f"postgresql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}")

    def ensure_table(self):
        sql = """CREATE TABLE IF NOT EXISTS yurumein_lexicon (id BIGSERIAL PRIMARY KEY, spanish TEXT, spanish_norm TEXT, garifuna TEXT NOT NULL, english TEXT, english_norm TEXT);"""

        with self.engine.begin() as conn:
            conn.execute(text(sql))

    def lookup(self, term: str, language: str) -> list:
        lang = language.lower()
        if lang not in _SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported lookup language: {language!r}. Supported: 'english', 'spanish'."
            )
        norm = normalize_term(term)
        column = "spanish_norm" if lang == "spanish" else "english_norm"
        sql = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sql, {"term": norm}).mappings().all()]
