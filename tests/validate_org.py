"""
SkillPath Phases 1 + 3 - validation script.

Proves the org layer and the adjacency engine on the sample Finance org, and
prints output a human can eyeball:

  1. ASSERT: the sample org loads with zero blocking issues and every SOC is rated.
  2. ASSERT: org_path (the naive reports_to walk) is the expected chain.
  3. ASSERT: the candidate ranking is well-formed (sorted, same-SOC rung at 1.0)
     and every senior finance role beats an unrelated control occupation. Whether
     the org's own order matches skill distance is REPORTED, not asserted - that
     disagreement is Phase 5's subject, not a bug.
  4. ASSERT: the ladder reaches the top and EVERY step recommends both a skill
     and a knowledge area (the split-gap decision).
  5. ASSERT: the same-SOC step is flagged and borrows recommendations rather
     than returning nothing (the same-occupation decision).

Exit code 0 = all assertions passed; non-zero = a check failed.

Run:  python tests/validate_org.py    (after the DB is built)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Make 'src' importable whether run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import build_ladder, format_ladder, rank_candidates  # noqa: E402
from src.org import errors, load_org, org_path, validate_org  # noqa: E402
from src.vectors import DB_PATH, is_rated, occupation_title, similarity  # noqa: E402

START_ROLE = "fin-analyst"
EXPECTED_ORG_PATH = ["fin-analyst", "fin-mgr", "fin-director"]
EXPECTED_TOP = "fin-director"
SAME_SOC_PAIR = ("fin-analyst", "fin-sr-analyst")

# Unrelated control from the spec: no finance role should look like a kitchen role.
UNRELATED_SOC = "35-2014.00"  # Cooks, Restaurant

BAR = "=" * 70


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def check_org_loads(org, conn) -> bool:
    print(BAR)
    print("CHECK 1 - sample org loads clean and every SOC has a vector")
    print(BAR)
    print(f"  {org.company}  |  function: {org.function}  |  {len(org.roles)} roles\n")

    print(f"  {'role_id':<16} {'lvl':>3} {'title':<28} {'SOC':<12} rated")
    print(f"  {'-'*16} {'-'*3} {'-'*28} {'-'*12} -----")
    for r in sorted(org.roles, key=lambda r: (r.level, r.role_id)):
        rated = "yes" if r.onet_soc and is_rated(r.onet_soc, conn) else "NO"
        print(f"  {r.role_id:<16} {r.level:>3} {r.title:<28.28} {r.onet_soc or '-':<12} {rated}")

    issues = validate_org(org, conn)
    blocking = errors(issues)
    print(f"\n  validation issues: {len(issues)} ({len(blocking)} blocking)")
    for i in issues:
        print(f"    {i}")

    passed = not blocking
    print(f"\n  ASSERT  zero blocking issues                   ->  [{_verdict(passed)}]")
    return passed


def check_org_path(org) -> bool:
    print("\n" + BAR)
    print("CHECK 2 - the org path (naive reports_to walk)")
    print(BAR)

    path = [r.role_id for r in org_path(org, START_ROLE)]
    titles = " -> ".join(r.title for r in org_path(org, START_ROLE))
    print(f"  {titles}\n")
    print(f"  got:      {path}")
    print(f"  expected: {EXPECTED_ORG_PATH}")

    passed = path == EXPECTED_ORG_PATH
    print(f"\n  ASSERT  org_path({START_ROLE}) matches expected  ->  [{_verdict(passed)}]")
    return passed


def check_adjacency_ordering(org, conn) -> bool:
    """
    CHECK 3 - the ranking is well-formed and the finance cluster is real.

    WHAT THIS DELIBERATELY DOES NOT ASSERT
    --------------------------------------
    An earlier version asserted sim(analyst, mgr) > sim(analyst, director) - that
    the org ladder's order matches skill distance. That is precisely the assumption
    SkillPath exists to *question*: Phase 5's divergence flag is built to surface
    orgs where the two disagree. Hard-coding the org order as ground truth would
    make the validation contradict the product's premise.

    It is also a knife-edge comparison. Filtering O*NET's 'Recommend Suppress'
    rows moved sim(analyst, mgr) from 0.852 to 0.821 while leaving the director
    pair at 0.838, flipping the order - because Financial Managers happened to have
    16 suppressed Level rows and Chief Executives had 1. That is an artifact of
    O*NET's reporting confidence, not a fact about finance careers.

    So we assert what is structurally true and report the rest as an observation.
    """
    print("\n" + BAR)
    print("CHECK 3 - the candidate ranking is well-formed and finance-clustered")
    print(BAR)

    ranked = rank_candidates(org, START_ROLE, conn)
    print(f"  Candidates for {org.get(START_ROLE).title}, most-adjacent first:\n")
    print(f"  {'cosine':>7}  {'role_id':<16} {'lvl':>3}  {'title':<28} same-SOC")
    print(f"  {'-'*7}  {'-'*16} {'-'*3}  {'-'*28} --------")
    for c in ranked:
        print(
            f"  {c.similarity:>7.4f}  {c.role.role_id:<16} {c.role.level:>3}  "
            f"{c.role.title:<28.28} {'yes' if c.same_occupation else ''}"
        )

    start_soc = org.get(START_ROLE).onet_soc
    control = similarity(start_soc, UNRELATED_SOC, conn)

    # (a) sorted descending, (b) the same-SOC rung is an exact 1.0 and ranks first,
    # (c) every senior finance role beats an unrelated occupation.
    sims = [c.similarity for c in ranked]
    sorted_ok = sims == sorted(sims, reverse=True)
    same_soc = [c for c in ranked if c.same_occupation]
    same_soc_ok = bool(same_soc) and same_soc[0] is ranked[0] and abs(sims[0] - 1.0) < 1e-9
    cluster_ok = all(s > control for s in sims)

    print(f"\n  Unrelated control: {occupation_title(UNRELATED_SOC, conn)} ({UNRELATED_SOC})")
    print(f"    sim(analyst, control) = {control:.4f}   worst candidate = {min(sims):.4f}")

    passed = sorted_ok and same_soc_ok and cluster_ok
    print(f"\n  ASSERT  ranked descending .................... [{_verdict(sorted_ok)}]")
    print(f"  ASSERT  same-SOC rung ranks first at 1.0000 .. [{_verdict(same_soc_ok)}]")
    print(f"  ASSERT  every candidate beats the control .... [{_verdict(cluster_ok)}]")

    # Observation, not an assertion - this is Phase 5's raw material.
    by_id = {c.role.role_id: c.similarity for c in ranked}
    mgr, director = by_id["fin-mgr"], by_id["fin-director"]
    verdict = "agree" if mgr > director else "DIVERGE"
    print(f"\n  OBSERVE org order vs skill order: {verdict}")
    print(f"          sim(analyst, mgr)      = {mgr:.4f}  (org: next rung up)")
    print(f"          sim(analyst, director) = {director:.4f}  (org: two rungs up)")
    if mgr <= director:
        print("          The org ladder and the skill ladder disagree here - exactly the")
        print("          case Phase 5's divergence flag is meant to surface, not hide.")
    return passed


def check_ladder(org, conn):
    """Returns (passed_ladder, passed_same_soc) - checks 4 and 5 share one ladder."""
    print("\n" + BAR)
    print("CHECK 4/5 - the skill-natural ladder and its recommendations")
    print(BAR)

    ladder = build_ladder(org, START_ROLE, conn)
    print(format_ladder(ladder, conn))

    # --- check 4: reaches the top, and every step has both kinds of rec ------ #
    rungs = [r.role_id for r in ladder.rungs]
    reached_top = ladder.top.role_id == EXPECTED_TOP

    print("\n  " + "-" * 66)
    print(f"  rungs: {' -> '.join(rungs)}")
    print(f"  {'step':<34} {'#skills':>8} {'#knowledge':>11}  recs from")
    print(f"  {'-'*34} {'-'*8} {'-'*11}  ---------")
    both_kinds = True
    for s in ladder.steps:
        n_sk = len(s.recommendations["skills"])
        n_kn = len(s.recommendations["knowledge"])
        if n_sk == 0 or n_kn == 0:
            both_kinds = False
        src = s.recommendations_from or "(none)"
        if s.substituted:
            src += "  <- substituted"
        label = f"{s.from_role.role_id} -> {s.to_role.role_id}"
        print(f"  {label:<34} {n_sk:>8} {n_kn:>11}  {src}")

    passed_ladder = reached_top and both_kinds
    print(f"\n  ASSERT  ladder tops out at '{EXPECTED_TOP}'          ->  [{_verdict(reached_top)}]")
    print(f"  ASSERT  every step recommends skills AND knowledge  ->  [{_verdict(both_kinds)}]")

    # --- check 5: the same-SOC step is handled, not swallowed ---------------- #
    frm, to = SAME_SOC_PAIR
    step = next(
        (s for s in ladder.steps if s.from_role.role_id == frm and s.to_role.role_id == to),
        None,
    )
    print("\n  " + "-" * 66)
    print(f"  Same-occupation step: {frm} -> {to}")
    if step is None:
        print("    step not found in the ladder!")
        passed_same = False
    else:
        has_recs = bool(step.recommendations["skills"] or step.recommendations["knowledge"])
        print(f"    same_occupation ....... {step.same_occupation}")
        print(f"    similarity ............ {step.similarity:.4f}")
        print(f"    recommendations_from .. {step.recommendations_from}")
        print(f"    substituted ........... {step.substituted}")
        print(f"    note .................. {step.note}")
        passed_same = (
            step.same_occupation
            and round(step.similarity, 4) == 1.0
            and step.recommendations_from == "fin-mgr"
            and has_recs
        )
    print(
        f"\n  ASSERT  same-SOC step flagged, sim 1.00, recs borrowed from "
        f"fin-mgr  ->  [{_verdict(passed_same)}]"
    )

    return passed_ladder, passed_same


def main() -> int:
    if not Path(DB_PATH).exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Build it first:  python build.py", file=sys.stderr)
        return 2

    conn = sqlite3.connect(DB_PATH)
    try:
        org = load_org()
        results = [
            ("org loads clean", check_org_loads(org, conn)),
            ("org path", check_org_path(org)),
            ("candidate ranking", check_adjacency_ordering(org, conn)),
        ]
        passed_ladder, passed_same = check_ladder(org, conn)
        results.append(("ladder + split recommendations", passed_ladder))
        results.append(("same-occupation handling", passed_same))
    finally:
        conn.close()

    print("\n" + BAR)
    failed = [name for name, ok in results if not ok]
    if not failed:
        print(f"RESULT: all {len(results)} Phase 1 + 3 assertions PASSED.")
        print(BAR)
        return 0
    print(f"RESULT: {len(failed)} assertion(s) FAILED: {', '.join(failed)}")
    print(BAR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
