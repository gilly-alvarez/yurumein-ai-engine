"""Session fixtures for integration tests that require the live translator service."""
import os
import pytest
import psycopg2


def _db_connect():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "db"),
        port=int(os.environ.get("DB_PORT", 5432)),
        dbname=os.environ.get("DB_NAME", "yurumein_test"),
        user=os.environ.get("DB_USER", "yurumein_test"),
        password=os.environ.get("DB_PASS", "yurumein_test"),
    )


@pytest.fixture(scope="session", autouse=True)
def seed_lexicon_test_rows():
    """Insert minimal deterministic lexicon rows required by TestMatchTypeIntegration.

    Rows use negative IDs to avoid conflict with real lexicon data.
    Cleaned up unconditionally in teardown — never persisted beyond the session.

    Requires migration 003 (quality_flag + garifuna_norm columns must exist).
    garifuna_norm is a GENERATED ALWAYS AS column and must not appear in INSERT.
    """
    try:
        conn = _db_connect()
    except Exception:
        # DB unreachable (e.g. running unit tests outside Docker): skip silently.
        yield
        return

    conn.autocommit = False
    cur = conn.cursor()
    try:
        # Row for EN→GAR exact match (test_exact_match_returns_match_type_exact).
        # english_norm='bracelet' → lookup_with_type("Bracelet", "en") hits tier 1.
        cur.execute("""
            INSERT INTO yurumein_lexicon
                (id, english, english_norm, garifuna, quality_flag)
            VALUES (-1, 'Bracelet', 'bracelet', 'BRACELET_FIXTURE', 'ok')
            ON CONFLICT (id) DO NOTHING
        """)
        # Row for GAR→EN lookups (test_folded_match and test_exact_garifuna_match).
        # garifuna='Wéndeti' → garifuna_norm auto-generated as 'wéndeti'.
        # Exact accented input 'Wéndeti' → tier 1 hit (match_type='exact').
        # Unaccented input 'Wendeti' → tier 1 misses, tier 2 folded scan hits.
        cur.execute("""
            INSERT INTO yurumein_lexicon
                (id, garifuna, english, english_norm, quality_flag)
            VALUES (-2, 'Wéndeti', 'WENDETI_FIXTURE', 'wendeti_fixture', 'ok')
            ON CONFLICT (id) DO NOTHING
        """)
        conn.commit()
        yield
    finally:
        cur.execute("DELETE FROM yurumein_lexicon WHERE id IN (-1, -2)")
        conn.commit()
        cur.close()
        conn.close()
