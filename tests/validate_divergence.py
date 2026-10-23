"""
SkillPath Phase 5 - validation script.

Proves the org-vs-skill divergence flag on the sample Finance org, and prints the
callout a human can eyeball:

  1. ASSERT: the Financial Analyst flags off-ladder occupations, every one of them
     outside the org, ranked, with a positive margin over the org's best step up.
  2. ASSERT: the baseline ignores same-occupation rungs. Senior Financial Analyst
     shares the analyst's SOC and scores exactly 1.0, so using it as the baseline
     would silence the flag permanently - the regression this check exists for.
  3. ASSERT: occupations sharing a BROAD SOC with an org role are excluded
     (11-3031.01 Treasurers is not "a track you have no ladder for" when the org
     already has 11-3031.00 Financial Managers).
  4. ASSERT: path divergence is detected - the org chart sends the analyst to
     Finance Manager, but Finance Director is the smaller skill jump.
  5. ASSERT: the top of the ladder flags nothing and does not crash (no senior
     roles means no baseline, so the flag would have no meaning).
  6. ASSERT: results are deterministic across repeated calls.

Exit code 0 = all assertions passed; non-zero = a check failed.

Run:  python tests/validate_divergence.py    (after the DB is built)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Make 'src' importable whether run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.divergence import (  # noqa: E402
    soc_base,
    divergence,
    format_divergence,
    off_ladder_matches,
    path_divergence,
)
from src.org import load_org  # noqa: E402
from src.vectors import DB_PATH  # noqa: E402

START_ROLE = "fin-analyst"
TOP_ROLE = "fin-director"
SAME_SOC_RUNG = "fin-sr-analyst"    # shares 13-2099.01 with START_ROLE
EXCLUDED_DETAIL_SOCS = ("11-3031.01", "11-3031.03")  # detail codes under Financial Managers

BAR = "=" * 70


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def check_off_ladder(org, conn) -> bool:
    print(BAR)
    print("CHECK 1 - the analyst's skills point outside the org, and it is ranked")
    print(BAR)

    div = divergence(org, START_ROLE, conn)
    matches = div.off_ladder
    org_bases = {soc_base(r.onet_soc) for r in org.roles if r.onet_soc}

    print(f"  Baseline (best step up the org offers): {div.baseline_role.title} "
          f"= {div.baseline_similarity:.4f}\n")
    print(f"  {'cosine':>7}  {'margin':>8}  soc          title")
    print(f"  {'-'*7}  {'-'*8}  {'-'*12} {'-'*40}")
    for m in matches:
        print(f"  {m.similarity:>7.4f}  {m.margin:>+8.4f}  {m.soc_code}  {m.title[:40]}")

    non_empty = bool(matches)
    all_outside = all(soc_base(m.soc_code) not in org_bases for m in matches)
    all_positive = all(m.margin > 0 for m in matches)
    sims = [m.similarity for m in matches]
    ranked = sims == sorted(sims, reverse=True)

    passed = non_empty and all_outside and all_positive and ranked
    print(f"\n  ASSERT  at least one off-ladder match ........ [{_verdict(non_empty)}]")
    print(f"  ASSERT  every match is outside the org ...... [{_verdict(all_outside)}]")
    print(f"  ASSERT  every margin beats the org baseline .. [{_verdict(all_positive)}]")
    print(f"  ASSERT  ranked by similarity ................. [{_verdict(ranked)}]")
    return passed


def check_baseline_ignores_same_soc(org, conn) -> bool:
    """
    The regression guard for the bug that made this flag silent.

    Senior Financial Analyst carries the same SOC as Financial Analyst, so its
    cosine is exactly 1.0. Taking 'the most similar senior role' at face value
    makes the baseline unbeatable and off_ladder_matches() always returns [].
    """
    print("\n" + BAR)
    print("CHECK 2 - the baseline ignores same-occupation rungs")
    print(BAR)

    div = divergence(org, START_ROLE, conn)
    same_soc_rung = org.get(SAME_SOC_RUNG)
    start = org.get(START_ROLE)

    print(f"  {start.title} SOC ............ {start.onet_soc}")
    print(f"  {same_soc_rung.title} SOC ..... {same_soc_rung.onet_soc}  (identical -> cosine 1.0)")
    print(f"  baseline actually used ....... {div.baseline_role.title} "
          f"({div.baseline_role.onet_soc}) = {div.baseline_similarity:.4f}")

    not_same_rung = div.baseline_role.role_id != SAME_SOC_RUNG
    not_saturated = div.baseline_similarity < 1.0
    still_flags = bool(div.off_ladder)

    passed = not_same_rung and not_saturated and still_flags
    print(f"\n  ASSERT  baseline is not the same-SOC rung .... [{_verdict(not_same_rung)}]")
    print(f"  ASSERT  baseline is below 1.0 ................ [{_verdict(not_saturated)}]")
    print(f"  ASSERT  the flag still fires ................. [{_verdict(still_flags)}]")
    return passed


def check_detail_socs_excluded(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 3 - detail codes of an occupation the org HAS are excluded")
    print(BAR)

    # Scan deep enough that the detail codes would surface if they were not excluded.
    deep = off_ladder_matches(org, START_ROLE, conn, top_n=50)
    found = [m.soc_code for m in deep if m.soc_code in EXCLUDED_DETAIL_SOCS]

    print(f"  The org has 11-3031.00 Financial Managers, so these detail codes")
    print(f"  are the same ladder and must not be reported as missing:")
    for soc in EXCLUDED_DETAIL_SOCS:
        print(f"    {soc}")
    print(f"\n  scanned top {len(deep)} matches; offending codes present: {found or 'none'}")

    passed = not found
    print(f"\n  ASSERT  no detail code of an org occupation .. [{_verdict(passed)}]")
    return passed


def check_path_divergence(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 4 - the org chart and the skill ladder disagree, and we say so")
    print(BAR)

    path = path_divergence(org, START_ROLE, conn)
    print(f"  org chart says next .... {path.org_next.title} "
          f"(cosine {path.org_next_similarity:.4f})")
    print(f"  skills say next ........ {path.skill_next.title} "
          f"(cosine {path.skill_next_similarity:.4f})")
    print(f"  agrees ................. {path.agrees}")
    print(f"  margin ................. {path.margin:+.4f}")

    detected = not path.agrees
    margin_ok = path.margin > 0
    both_present = path.comparable

    passed = detected and margin_ok and both_present
    print(f"\n  ASSERT  divergence detected .................. [{_verdict(detected)}]")
    print(f"  ASSERT  margin is positive ................... [{_verdict(margin_ok)}]")
    print(f"  ASSERT  both sides resolved .................. [{_verdict(both_present)}]")
    return passed


def check_top_of_ladder(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 5 - the top of the ladder flags nothing rather than crashing")
    print(BAR)

    div = divergence(org, TOP_ROLE, conn)
    print(f"  role ............... {div.role.title}")
    print(f"  baseline_role ...... {div.baseline_role}")
    print(f"  off_ladder ......... {len(div.off_ladder)} matches")
    print(f"  path.comparable .... {div.path.comparable}")
    print(f"  flagged ............ {div.flagged}")

    passed = (
        div.baseline_role is None
        and div.off_ladder == []
        and div.path.agrees
        and not div.flagged
    )
    print(f"\n  ASSERT  no baseline, no matches, no flag ..... [{_verdict(passed)}]")
    return passed


def check_deterministic(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 6 - repeated calls return identical results")
    print(BAR)

    a = divergence(org, START_ROLE, conn)
    b = divergence(org, START_ROLE, conn)
    same = (
        [(m.soc_code, m.similarity, m.margin) for m in a.off_ladder]
        == [(m.soc_code, m.similarity, m.margin) for m in b.off_ladder]
        and a.baseline_similarity == b.baseline_similarity
        and a.path.skill_next.role_id == b.path.skill_next.role_id
    )
    print(f"  two independent divergence() calls agree: {same}")
    print(f"\n  ASSERT  deterministic ........................ [{_verdict(same)}]")
    return same


def show_callout(org, conn) -> None:
    print("\n" + BAR)
    print("THE CALLOUT (spec section 7 'money shot')")
    print(BAR)
    print(format_divergence(divergence(org, START_ROLE, conn), conn))


def main() -> int:
    if not Path(DB_PATH).exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Build it first:  python build.py", file=sys.stderr)
        return 2

    conn = sqlite3.connect(DB_PATH)
    try:
        org = load_org()
        results = [
            ("off-ladder matches", check_off_ladder(org, conn)),
            ("baseline ignores same-SOC", check_baseline_ignores_same_soc(org, conn)),
            ("detail-SOC exclusion", check_detail_socs_excluded(org, conn)),
            ("path divergence", check_path_divergence(org, conn)),
            ("top of ladder", check_top_of_ladder(org, conn)),
            ("determinism", check_deterministic(org, conn)),
        ]
        show_callout(org, conn)
    finally:
        conn.close()

    print("\n" + BAR)
    failed = [name for name, ok in results if not ok]
    if not failed:
        print(f"RESULT: all {len(results)} Phase 5 assertions PASSED.")
        print(BAR)
        return 0
    print(f"RESULT: {len(failed)} assertion(s) FAILED: {', '.join(failed)}")
    print(BAR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
