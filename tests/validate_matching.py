"""
SkillPath Phases 2 + 4 - validation script.

Covers the DETERMINISTIC halves of the matching and UI layers. Deliberately makes
**no API calls**, so `build.py` stays runnable without an ANTHROPIC_API_KEY and in
CI. What is checked:

  1. ASSERT: the shortlist for "Financial Analyst" contains both plausible codes.
  2. ASSERT: the searches that the first word-matching implementation got wrong
     now return the right occupation ("Accounts Payable Clerk", "Staff Accountant").
  3. ASSERT: every hit is a RATED occupation - the guard that stops an unrated
     code ever reaching the math layer.
  4. ASSERT: _validate_pick rewrites a hallucinated SOC and downgrades confidence.
  5. ASSERT: ladder_dot emits one node per rung and one edge per step, dashing
     the same-occupation edge.

What is NOT covered here, by design: whether Claude picks well, and whether the
Streamlit app renders. The first needs the API, the second needs a browser.

Exit code 0 = all assertions passed; non-zero = a check failed.

Run:  python tests/validate_matching.py    (after the DB is built)
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import build_ladder, ladder_dot  # noqa: E402
from src.matching import (  # noqa: E402
    ClarifyingQuestion,
    MatchProposal,
    _validate_pick,
    search_occupations,
)
from src.org import load_org  # noqa: E402
from src.vectors import DB_PATH, is_rated  # noqa: E402

BAR = "=" * 70

# 13-2051.00 (Financial and Investment Analysts) is unrated, so the two defensible
# stand-ins for a corporate financial analyst are the quant code and the budget code.
ANALYST_EXPECTED = {"13-2099.01", "13-2031.00"}

# Searches the original word-overlap implementation ranked wrongly.
REGRESSION_SEARCHES = [
    ("Accounts Payable Clerk", "43-3031.00", False),  # must appear in the shortlist
    ("Staff Accountant", "13-2011.00", True),         # must rank FIRST
    ("Finance Manager", "11-3031.00", True),
]


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def check_analyst_shortlist() -> bool:
    print(BAR)
    print("CHECK 1 - the 'Financial Analyst' shortlist holds both plausible codes")
    print(BAR)

    hits = search_occupations("Financial Analyst", limit=12)
    codes = [h.soc_code for h in hits]
    print(f"  {'score':>7}  {'SOC':<12} title")
    print(f"  {'-'*7}  {'-'*12} {'-'*40}")
    for h in hits[:8]:
        print(f"  {h.score:>7.1f}  {h.soc_code:<12} {h.title}")
    if len(hits) > 8:
        print(f"  ... and {len(hits) - 8} more")

    missing = ANALYST_EXPECTED - set(codes)
    passed = not missing
    print(f"\n  looking for: {sorted(ANALYST_EXPECTED)}")
    if missing:
        print(f"  MISSING:     {sorted(missing)}")
    print(f"\n  ASSERT  both candidate codes are on the shortlist  ->  [{_verdict(passed)}]")
    return passed


def check_regression_searches() -> bool:
    print("\n" + BAR)
    print("CHECK 2 - searches the first word-matching implementation got wrong")
    print(BAR)
    print("  (it missed Finance/Financial, Analyst/Analysts, Accounts/Accounting)\n")

    print(f"  {'query':<26} {'wanted':<12} {'rank':>5}  requirement")
    print(f"  {'-'*26} {'-'*12} {'-'*5}  {'-'*22}")
    passed = True
    for query, wanted, must_be_first in REGRESSION_SEARCHES:
        codes = [h.soc_code for h in search_occupations(query, limit=12)]
        rank = codes.index(wanted) + 1 if wanted in codes else 0
        ok = rank == 1 if must_be_first else rank > 0
        passed = passed and ok
        requirement = "must rank first" if must_be_first else "must be shortlisted"
        shown = str(rank) if rank else "ABSENT"
        print(f"  {query:<26} {wanted:<12} {shown:>5}  {requirement}  [{_verdict(ok)}]")

    print(f"\n  ASSERT  every regression search resolves  ->  [{_verdict(passed)}]")
    return passed


def check_all_hits_rated(conn) -> bool:
    print("\n" + BAR)
    print("CHECK 3 - every search hit is a RATED occupation")
    print(BAR)
    print("  This is the guard that keeps an unrated SOC out of the math layer:")
    print("  build_vector() raises on one, so the shortlist must never offer one.\n")

    queries = ["Financial Analyst", "Accountant", "Clerk", "Manager", "Director", "Engineer"]
    unrated: list[tuple[str, str]] = []
    total = 0
    for q in queries:
        hits = search_occupations(q, limit=12)
        total += len(hits)
        for h in hits:
            if not is_rated(h.soc_code, conn):
                unrated.append((q, h.soc_code))
        print(f"  {q:<20} {len(hits):>3} hits, all rated: {not any(u[0] == q for u in unrated)}")

    passed = not unrated
    print(f"\n  {total} hits checked across {len(queries)} queries; {len(unrated)} unrated")
    for q, soc in unrated[:5]:
        print(f"    UNRATED: {soc} returned for '{q}'")
    print(f"\n  ASSERT  no unrated occupation is ever shortlisted  ->  [{_verdict(passed)}]")
    return passed


def check_hallucination_guard() -> bool:
    print("\n" + BAR)
    print("CHECK 4 - a SOC that is not on the shortlist is rejected")
    print(BAR)
    print("  Simulates Claude returning a plausible-looking code it was never offered.")
    print("  No API call - the proposal is constructed by hand.\n")

    candidates = search_occupations("Financial Analyst", limit=12)
    bogus = "99-9999.00"
    assert bogus not in {h.soc_code for h in candidates}

    proposal = MatchProposal(
        soc_code=bogus,
        confidence="high",
        reasoning="Fabricated for the test.",
        runner_up_soc="88-8888.00",
        questions=[
            ClarifyingQuestion(question="q1?", why="because", options=["a", "b"]),
            ClarifyingQuestion(question="q2?", why="because", options=["a", "b"]),
        ],
    )
    warning = _validate_pick(proposal, candidates)

    top_hit = candidates[0].soc_code
    print(f"  proposed:          {bogus}  (confidence 'high')")
    print(f"  after the guard:   {proposal.soc_code}  (confidence '{proposal.confidence}')")
    print(f"  top keyword hit:   {top_hit}")
    print(f"  runner_up_soc:     {proposal.runner_up_soc}  (bogus one should be cleared)")
    print(f"  warning raised:    {warning}")

    passed = (
        proposal.soc_code == top_hit
        and proposal.confidence == "low"
        and proposal.runner_up_soc is None
        and warning is not None
    )
    print(
        f"\n  ASSERT  bogus SOC replaced, confidence downgraded, user warned  "
        f"->  [{_verdict(passed)}]"
    )
    return passed


def check_ladder_dot(conn) -> bool:
    print("\n" + BAR)
    print("CHECK 5 - ladder_dot emits well-formed DOT for the UI")
    print(BAR)

    org = load_org()
    ladder = build_ladder(org, "fin-analyst", conn)
    dot = ladder_dot(ladder, conn)
    print(dot)

    nodes = re.findall(r"^\s+r(\d+) \[label=", dot, flags=re.MULTILINE)
    edges = re.findall(r"^\s+r(\d+) -> r(\d+) \[", dot, flags=re.MULTILINE)
    dashed = re.findall(r"^\s+r\d+ -> r\d+ \[.*style=dashed\]", dot, flags=re.MULTILINE)
    same_occ = [s for s in ladder.steps if s.same_occupation]

    print(f"\n  rungs: {len(ladder.rungs)}   nodes emitted: {len(nodes)}")
    print(f"  steps: {len(ladder.steps)}   edges emitted: {len(edges)}")
    print(f"  same-occupation steps: {len(same_occ)}   dashed edges: {len(dashed)}")

    passed = (
        dot.startswith("digraph ladder {")
        and dot.rstrip().endswith("}")
        and len(nodes) == len(ladder.rungs)
        and len(edges) == len(ladder.steps)
        and len(dashed) == len(same_occ)
    )
    print(f"\n  ASSERT  one node per rung, one edge per step, dashed = same-SOC  "
          f"->  [{_verdict(passed)}]")
    return passed


def main() -> int:
    if not Path(DB_PATH).exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Build it first:  python build.py", file=sys.stderr)
        return 2

    conn = sqlite3.connect(DB_PATH)
    try:
        results = [
            ("analyst shortlist", check_analyst_shortlist()),
            ("regression searches", check_regression_searches()),
            ("all hits rated", check_all_hits_rated(conn)),
            ("hallucination guard", check_hallucination_guard()),
            ("ladder DOT", check_ladder_dot(conn)),
        ]
    finally:
        conn.close()

    print("\n" + BAR)
    failed = [name for name, ok in results if not ok]
    if not failed:
        print(f"RESULT: all {len(results)} Phase 2 + 4 assertions PASSED (no API calls made).")
        print(BAR)
        return 0
    print(f"RESULT: {len(failed)} assertion(s) FAILED: {', '.join(failed)}")
    print(BAR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
