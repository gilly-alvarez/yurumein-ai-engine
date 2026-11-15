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
        """EN/ES -> Garifuna lookup by normalized column.

        For English queries, if exact match fails and the query has no
        parenthetical annotation, retries with a prefix match to find entries
        like 'handsome (masculine)' when the user typed 'handsome'.
        """
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang in {"spanish", "es"} else "english_norm"
        sql_exact = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql_exact, {"term": norm}).mappings().all()]
            if rows or column != "english_norm" or "(" in norm:
                return rows
            # Annotation-stripped fallback: match english_norm values that begin
            # with the query followed by a space and open-paren, e.g.:
            #   "handsome" matches "handsome (masculine)", "handsome (feminine)"
            sql_prefix = text(
                "SELECT * FROM yurumein_lexicon WHERE english_norm LIKE :prefix"
            )
            return [dict(r) for r in conn.execute(sql_prefix, {"prefix": norm + " (%"}).mappings().all()]

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
