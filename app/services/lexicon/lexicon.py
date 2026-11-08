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
            f"postgresql://{settings.DB_USER}:{settings.DB_PASS}"
            f"@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"
        )

    def lookup(self, term: str, language: str) -> list:
        """EN/ES -> Garifuna lookup by normalized column."""
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang == "spanish" else "english_norm"
        sql = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            return [dict(r) for r in conn.execute(sql, {"term": norm}).mappings().all()]

    def reverse_lookup(self, term: str) -> list:
        """Exact normalized match, handling accented initials and
        comma-separated variant forms. No substring matching."""
        norm = normalize_term(term)
        if not norm:
            return []
        sql = text(
            "SELECT * FROM yurumein_lexicon WHERE garifuna IS NOT NULL"
        )
        with self.engine.connect() as conn:
            candidates = [
                dict(r) for r in conn.execute(sql).mappings().all()
            ]
        result = []
        for row in candidates:
            forms = [f.strip() for f in (row.get("garifuna") or "").split(",")]
            if any(normalize_term(f) == norm for f in forms):
                result.append(row)
        return result
