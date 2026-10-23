"""
SkillPath Phase 0 - validation script.

Proves the skill-adjacency math is sane on transitions where we already know
the answer, and prints output a human can eyeball:

  1. ASSERT: Financial Analyst is MORE similar to Financial Managers than to an
     unrelated role (Chefs & Head Cooks).
  2. PRINT : top-5 skill_gap for Financial Analyst -> Financial Managers
     (expect management / administration skills to rise).
  3. PRINT : a similarity matrix across several finance-related roles.

Exit code 0 = all assertions passed; non-zero = a check failed.

Run:  python tests/validate.py    (after the DB is built)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Make 'src' importable whether run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.vectors import DB_PATH, build_vector, occupation_title, similarity, skill_gap  # noqa: E402

# --- Roles used in the checks --------------------------------------------- #
# NOTE ON THE ANALYST CODE
# The spec's headline "Financial Analyst" is O*NET 13-2051.00 (Financial and
# Investment Analysts). O*NET 30.3 publishes NO skill/knowledge ratings for that
# code (data collection for the merged occupation is still pending), so it has no
# vector to compute on. We stand in the closest rated "financial analyst"
# occupation, 13-2099.01 Financial Quantitative Analysts, for the validation.
# See README ("O*NET 30.3 'Skills' restructure" and data notes) for detail.
FINANCIAL_ANALYST = "13-2099.01"   # Financial Quantitative Analysts (proxy for 13-2051.00, unrated)
SPEC_ANALYST_UNRATED = "13-2051.00"  # Financial and Investment Analysts (no O*NET 30.3 ratings)
FINANCIAL_MANAGERS = "11-3031.00"  # Financial Managers
CHEFS = "35-1011.00"               # Chefs and Head Cooks  (unrelated control)

FINANCE_ROLES = [
    "13-2099.01",  # Financial Quantitative Analysts (proxy; 13-2051.00 is unrated)
    "11-3031.00",  # Financial Managers
    "13-2011.00",  # Accountants and Auditors
    "13-1111.00",  # Management Analysts
    "13-2052.00",  # Personal Financial Advisors
]

BAR = "=" * 70


def _title(conn, soc):
    return f"{occupation_title(soc, conn)} ({soc})"


def check_similarity_ordering(conn) -> bool:
    print(BAR)
    print("CHECK 1 - Financial Analyst is closer to Finance Mgmt than to Chefs")
    print(BAR)
    print(f"  NOTE: spec's 13-2051.00 has no O*NET 30.3 ratings; using rated proxy")
    print(f"        {_title(conn, FINANCIAL_ANALYST)} as 'Financial Analyst'.\n")

    sim_related = similarity(FINANCIAL_ANALYST, FINANCIAL_MANAGERS, conn)
    sim_unrelated = similarity(FINANCIAL_ANALYST, CHEFS, conn)

    print(f"  {_title(conn, FINANCIAL_ANALYST)}")
    print(f"    -> {_title(conn, FINANCIAL_MANAGERS)}:  cosine = {sim_related:.4f}")
    print(f"    -> {_title(conn, CHEFS)}:  cosine = {sim_unrelated:.4f}")

    passed = sim_related > sim_unrelated
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  ASSERT  sim(FA, Finance Mgr) > sim(FA, Chefs)   ->  [{verdict}]")
    print(f"          {sim_related:.4f} > {sim_unrelated:.4f}  ==  {passed}")
    return passed


def show_skill_gap(conn) -> None:
    print("\n" + BAR)
    print("CHECK 2 - Top-5 skills to develop:  Financial Analyst -> Financial Managers")
    print(BAR)
    print(f"  From: {_title(conn, FINANCIAL_ANALYST)}")
    print(f"  To:   {_title(conn, FINANCIAL_MANAGERS)}")
    print("  (delta = LV_next - LV_current, positive only, top 5)\n")

    gaps = skill_gap(FINANCIAL_ANALYST, FINANCIAL_MANAGERS, conn, top_n=5)

    print(f"  {'#':<2} {'Descriptor':<38} {'type':<9} {'cur':>5} {'next':>5} {'delta':>6}")
    print(f"  {'-'*2} {'-'*38} {'-'*9} {'-'*5} {'-'*5} {'-'*6}")
    for g in gaps:
        print(
            f"  {g['rank']:<2} {g['element_name']:<38.38} {g['descriptor_type']:<9} "
            f"{g['lv_current']:>5.2f} {g['lv_next']:>5.2f} {g['delta']:>+6.2f}"
        )
    print("\n  (Eyeball check: expect management / administration / coordination to rise.)")


def show_similarity_matrix(conn) -> None:
    print("\n" + BAR)
    print("CHECK 3 - Similarity matrix across finance-related roles")
    print(BAR)

    labels = {soc: occupation_title(soc, conn) for soc in FINANCE_ROLES}
    for soc in FINANCE_ROLES:
        print(f"  {soc}  {labels[soc]}")
    print()

    # Short, unambiguous column headers = the 4-digit occupation group
    # (chars after the '-', before the '.'), e.g. '13-2011.00' -> '2011'.
    def short(soc: str) -> str:
        return soc.split("-", 1)[1].split(".", 1)[0]

    col_w = 8
    header = " " * 14 + "".join(f"{short(soc):>{col_w}}" for soc in FINANCE_ROLES)
    print(header)
    for a in FINANCE_ROLES:
        cells = "".join(f"{similarity(a, b, conn):>{col_w}.3f}" for b in FINANCE_ROLES)
        print(f"  {a:<12}{cells}")
    print("\n  (Diagonal is 1.000; finance roles should cluster high off-diagonal.)")


def main() -> int:
    if not Path(DB_PATH).exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Build it first:  python build.py   (or python -m src.onet_loader)", file=sys.stderr)
        return 2

    conn = sqlite3.connect(DB_PATH)
    try:
        # Sanity: vector dimensionality.
        dim = len(build_vector(FINANCIAL_ANALYST, conn))
        print(f"Vector space: {dim} descriptors (skills + knowledge)\n")

        passed = check_similarity_ordering(conn)
        show_skill_gap(conn)
        show_similarity_matrix(conn)
    finally:
        conn.close()

    print("\n" + BAR)
    if passed:
        print("RESULT: all assertions PASSED.")
        print(BAR)
        return 0
    print("RESULT: assertion FAILED - the math ordering is wrong, investigate.")
    print(BAR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
