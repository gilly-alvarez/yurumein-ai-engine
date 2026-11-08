import re
import unicodedata
from sqlalchemy import create_engine, text
from app.core.config import settings


def normalize_term(value: str) -> str:
    value = unicodedata.normalize("NFD", value or "")
    value = "".join(c for c in value if unicodedata.category(c) != "Mn")
    value = unicodedata.normalize("NFKC", value)
    return re.sub(r"\s+", " ", value.strip().lower())


class YurumeinLexicon:
    def __init__(self):
        self.engine = create_engine(
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )

    def lookup(self, term: str, language: str) -> list:
        """EN/ES → Garífuna lookup by normalized column."""
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang == "spanish" else "english_norm"
        sql = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sql, {"term": norm}).mappings().all()]

    def reverse_lookup(self, term: str) -> list:
        """
        Garífuna → EN/ES lookup.

        Tries exact case-insensitive match first, then substring match.
        Used for definition queries ("What does X mean?").
        """
        norm = normalize_term(term)
        if not norm:
            return []
        sql_exact = text(
            "SELECT * FROM yurumein_lexicon WHERE lower(garifuna) = :term LIMIT 10"
        )
        sql_contains = text(
            "SELECT * FROM yurumein_lexicon WHERE lower(garifuna) LIKE :term LIMIT 10"
        )
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql_exact, {"term": norm}).mappings().all()]
            if not rows:
                rows = [
                    dict(r)
                    for r in conn.execute(sql_contains, {"term": f"%{norm}%"}).mappings().all()
                ]
        return rows
