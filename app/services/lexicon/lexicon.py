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
        Garífuna → EN/ES lookup by exact normalized form only.

        Matches complete Garífuna words, not substrings. Both the query term
        and the stored garifuna value are run through normalize_term() before
        comparison to handle accent/case variants.

        Strategy: fetch all stored entries whose first character matches
        (narrowing the scan), then filter in Python by normalized equality.
        Returns empty when the term is not an exact stored form.
        """
        norm = normalize_term(term)
        if not norm:
            return []

        # Pull rows whose garifuna value starts with the same first character
        # (after lowercasing). This avoids a full-table scan while still
        # handling accent variants that survive lowercase but not normalize_term.
        prefix = norm[0]
        sql = text(
            "SELECT * FROM yurumein_lexicon "
            "WHERE lower(garifuna) LIKE :prefix "
            "LIMIT 500"
        )
        with self.engine.connect() as conn:
            candidates = [
                dict(r)
                for r in conn.execute(sql, {"prefix": f"{prefix}%"}).mappings().all()
            ]

        # Exact match only: normalize the stored form and compare.
        return [
            row for row in candidates
            if normalize_term(row.get("garifuna", "")) == norm
        ]
