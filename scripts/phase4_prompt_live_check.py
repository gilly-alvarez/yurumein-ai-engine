#!/usr/bin/env python3
"""
Phase 4 — Prompt Live Check
============================
Runs two targeted LLM calls with the new system prompt (RP01–RP06 added)
and manually-constructed retrieval context matching production RAG output.

Checks:
  Q1 — "What is the Garífuna word for father?"
       Expected: Baba, Tata, Úguchilin from Diccionario p.144; Pa from
                 Libro Serie I p.53 if included; NO informal/formal framing.

  Q2 — "When did the Garífuna arrive in Guatemala?"
       Expected: Enríquez Bermúdez attribution preserved; 1802 arrival
                 presented as documented history; specific details (Nov 25,
                 150 people, Díaz affiliation) flagged as single-source.

Usage (from project root):
  python scripts/phase4_prompt_live_check.py
"""
import os, sys, textwrap
from pathlib import Path

env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed.")
    sys.exit(1)

sys.path.insert(0, str(Path(__file__).parent.parent))
from app.services.utils.RAG_pipeline.prompt.prompt import system_prompt, user_prompt

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("MODEL_NAME", "gpt-4o")

# ---------------------------------------------------------------------------
# Retrieval context fixtures
# ---------------------------------------------------------------------------

# Q1: father — lexicon miss + Diccionario p.144 + Libro Serie I p.53
CONTEXT_FATHER = """\
<LEXICON_MISS>
No entry found for "father" in the Yurumein structured lexicon.
Checking approved documentary sources.
</LEXICON_MISS>

[Doc 1 | File: DICCIONARIO GARIFUNA.pdf | Canonical: DIMUREI AGEI - Enriquez | Page: 144 | Tier: B | Domain-authority lexical: A]
Padecer. V. trans. Áfaga magürabu
Padre. Adj. cal. Baba, Tata, Úguchilin
Padre Dios. Sus. com. Bungiu Úguchilin
Pagar. V. trans. Afayeiha, Afayei

[Doc 2 | File: LIBRO GARIFUNA SERIE I.pdf | Canonical: FURENDEI WAMÁ ABÜRÜHA — Garüdia Número Aban | Page: 53 | Tier: B | Domain-authority lexical: C]
PALABRAS MONOSILABAS EN IDIOMA GARIFUNA
Pa    Pa (Papá)
Pe    Pecho
Pi    Piel
"""

# Q2: Guatemala arrival — Libro Garifuna II p.5 (C004 account)
CONTEXT_GUATEMALA = """\
[Doc 1 | File: LIBRO GARIFUNA II.pdf | Canonical: FURENDEI WAMÁ AYANUHA — Garüdia Número Bíama | Page: 5 | Tier: B | Author: Elmer Mauricio Enríquez Bermúdez (ABÜRÜHATI)]
El 25 de noviembre de 1802, exactamente 150 miembros de Caribes Negros y Caribes Rojos — dirigidos por el militar Marcos Sánchez Díaz — llegaron a las costas de Guatemala. Esta fecha es reconocida actualmente como el Día de la llegada del Pueblo Garífuna a Guatemala.

[Doc 2 | File: LIBRO GARIFUNA II.pdf | Canonical: FURENDEI WAMÁ AYANUHA — Garüdia Número Bíama | Page: 4 | Tier: B]
Después de la deportación de 1797, los Garífunas fueron llevados a la isla de Roatán, Honduras. Desde allí, grupos de Garífunas se trasladaron gradualmente hacia las costas de América Central.
"""

# ---------------------------------------------------------------------------
# Acceptance checks
# ---------------------------------------------------------------------------

def check_father(response: str) -> tuple[bool, list[str]]:
    failures = []
    r = response.lower()
    # Must include the Diccionario forms
    if "baba" not in r:
        failures.append("MISSING: Baba not in response")
    if "tata" not in r:
        failures.append("MISSING: Tata not in response")
    if "úguchilin" not in r and "uguchilin" not in r and "uguchili" not in r:
        failures.append("MISSING: Úguchilin (or variant) not in response")
    # Must NOT invent informal/formal register framing
    if "informal" in r:
        failures.append("FAIL: Response contains 'informal' — unsupported register framing")
    if "formal" in r:
        failures.append("FAIL: Response contains 'formal' — unsupported register framing")
    # If Pa appears, must attribute to Libro Serie I (not as primary/only form)
    if " pa " in r or r.startswith("pa ") or '"pa"' in r or "'pa'" in r:
        if "serie i" not in r and "libro" not in r and "serie" not in r:
            failures.append("WARN: Pa mentioned without Libro Serie I attribution")
    return len(failures) == 0, failures


def check_guatemala(response: str) -> tuple[bool, list[str]]:
    failures = []
    r = response.lower()
    # Must preserve community author attribution
    if "enríquez" not in r and "enriquez" not in r and "abürühati" not in r:
        failures.append("FAIL: No attribution to Enríquez Bermúdez or ABÜRÜHATI")
    # 1802 arrival must be present
    if "1802" not in r:
        failures.append("FAIL: 1802 arrival date not mentioned")
    # Must not present specific details as plain fact without qualification
    # Either November 25 is absent, OR if present it must be hedged
    if "november 25" in r or "25 de noviembre" in r or "noviembre" in r:
        hedges = ["single", "account", "source", "attributed", "according", "states", "records",
                  "one source", "this account", "not independently", "without independent"]
        if not any(h in r for h in hedges):
            failures.append("FAIL: Specific date (Nov 25) stated without attribution hedge")
    return len(failures) == 0, failures


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_check(label: str, query: str, context: str, check_fn) -> bool:
    print(f"\n{'='*70}")
    print(f"CHECK: {label}")
    print(f"QUERY: {query}")
    print("=" * 70)

    filled_user = user_prompt.format(
        chat_history="",
        context=context,
        query=query,
    )

    resp = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": filled_user},
        ],
    )
    answer = resp.choices[0].message.content.strip()
    print(f"\nRESPONSE:\n{textwrap.indent(answer, '  ')}")

    passed, failures = check_fn(answer)
    print(f"\nVALIDATION: {'PASS' if passed else 'FAIL'}")
    for f in failures:
        print(f"  {f}")
    return passed


if __name__ == "__main__":
    p1 = run_check(
        "Q1 — Garífuna word for father",
        "What is the Garífuna word for father?",
        CONTEXT_FATHER,
        check_father,
    )
    p2 = run_check(
        "Q2 — Garífuna arrival in Guatemala",
        "When did the Garífuna arrive in Guatemala?",
        CONTEXT_GUATEMALA,
        check_guatemala,
    )

    print(f"\n{'='*70}")
    if p1 and p2:
        print("RESULT: ALL CHECKS PASS — approved for deployment.")
    else:
        print("RESULT: ONE OR MORE CHECKS FAILED — do not deploy.")
        sys.exit(1)
