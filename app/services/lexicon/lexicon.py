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

        For English queries, if exact match fails and the query contains no
        parenthetical annotation, retries with LIKE in two passes:

        Fix A — space-paren: matches '<base> (<annotation>)' e.g.
          "handsome" → "handsome (masculine)", "handsome (feminine)"
          Covers ~2,028 annotated entries stored with a space before '('.

        Fix B — comma-paren: if Fix A also misses, matches '<base>, (<annotation>)'
          e.g. "count" → "Count, (numbers)", "thin" → "Thin, (you)"
          Covers ~93 entries stored with a comma before the parenthetical.

        LIKE metacharacters (%, _, !) in the input are escaped so a query
        such as '50%' or 'some_thing' cannot act as a SQL wildcard.

        Scope:
          - Fix A fires when the base form has no exact entry and no annotated
            twin with the same exact normalized form.
          - Fix B fires only when Fix A also returns nothing.
          - When an exact entry for the base form exists, neither fallback is
            invoked; the exact match is returned and annotated variants are
            not appended.
        """
        lang = language.lower()
        norm = normalize_term(term)
        column = "spanish_norm" if lang in {"spanish", "es"} else "english_norm"
        sql_exact = text(f"SELECT * FROM yurumein_lexicon WHERE {column} = :term")
        with self.engine.connect() as conn:
            rows = [dict(r) for r in conn.execute(sql_exact, {"term": norm}).mappings().all()]
            if rows or column != "english_norm" or "(" in norm:
                return rows
            escaped = norm.replace("!", "!!").replace("%", "!%").replace("_", "!_")
            sql_like = text(
                "SELECT * FROM yurumein_lexicon"
                " WHERE english_norm LIKE :prefix ESCAPE '!'"
            )
            # Fix A: space-paren — '<base> (<annotation>)'
            rows = [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + " (%"}).mappings().all()]
            if rows:
                return rows
            # Fix B: comma-paren — '<base>, (<annotation>)'
            return [dict(r) for r in conn.execute(sql_like, {"prefix": escaped + ", (%"}).mappings().all()]

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
