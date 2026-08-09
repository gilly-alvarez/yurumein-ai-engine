#!/usr/bin/env python3
"""
Idempotent, additive-only loader for the yurumein_lexicon table.

Reads a CSV (default: yurumein_lexicon_469.csv at repo root) and inserts
rows that do not already exist. Existing rows are never modified or removed.

NOTE ON VERIFICATION STATUS:
These entries represent the documented lexicon from the Enríquez source
dataset. Row-level independent verification governance is not yet
represented in the current schema and will be addressed in a later
controlled schema migration. Exact matches returned by the translator are
authoritative relative to this imported source, not independently verified
entries.

Usage:
    python scripts/load_lexicon.py
    python scripts/load_lexicon.py --csv /path/to/other.csv
"""

import argparse
import csv
import sys
from pathlib import Path

# Ensure repo root is on sys.path so app modules are importable when the
# script is invoked directly (e.g. `python scripts/load_lexicon.py`).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text

from app.services.lexicon.lexicon import YurumeinLexicon, normalize_term

_DEFAULT_CSV = Path(__file__).resolve().parent.parent / "yurumein_lexicon_469.csv"

# WHERE NOT EXISTS enforces idempotency without a UNIQUE constraint: a row is
# inserted only when no existing row matches on (spanish_norm, garifuna,
# english_norm). Running the script twice on the same CSV is safe.
_INSERT_SQL = text("""
    INSERT INTO yurumein_lexicon (spanish, spanish_norm, garifuna, english, english_norm)
    SELECT :spanish, :spanish_norm, :garifuna, :english, :english_norm
    WHERE NOT EXISTS (
        SELECT 1 FROM yurumein_lexicon
        WHERE spanish_norm = :spanish_norm
          AND garifuna     = :garifuna
          AND english_norm = :english_norm
    )
""")


def load(csv_path: Path) -> None:
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    lexicon = YurumeinLexicon()
    lexicon.ensure_table()

    attempted = inserted = skipped = errors = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            attempted += 1
            garifuna = (row.get("garifuna") or "").strip()
            if not garifuna:
                skipped += 1
                continue
            try:
                params = {
                    "spanish":      (row.get("spanish") or "").strip() or None,
                    "spanish_norm": normalize_term(row.get("spanish") or ""),
                    "garifuna":     garifuna,
                    "english":      (row.get("english") or "").strip() or None,
                    "english_norm": normalize_term(row.get("english") or ""),
                }
                with lexicon.engine.begin() as conn:
                    result = conn.execute(_INSERT_SQL, params)
                if result.rowcount == 1:
                    inserted += 1
                else:
                    skipped += 1
            except Exception as exc:
                errors += 1
                print(f"  Row {attempted}: error — {exc}", file=sys.stderr)

    print(
        f"\nLoad complete — {attempted} rows attempted, "
        f"{inserted} inserted, {skipped} skipped, {errors} errors"
    )
    if errors:
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Additive, idempotent loader for yurumein_lexicon table.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=_DEFAULT_CSV,
        help=f"Path to source CSV (default: {_DEFAULT_CSV})",
    )
    args = parser.parse_args()
    load(args.csv)


if __name__ == "__main__":
    main()
