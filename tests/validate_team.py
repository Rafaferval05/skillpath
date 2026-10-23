"""
SkillPath Phase 6 - validation script.

Proves the team-level drift view on the sample Finance org, and prints the
department callout a human can eyeball:

  1. ASSERT: every role appears exactly once and the people add up - the view
     accounts for the whole function, not just the roles it can assess.
  2. ASSERT: the aggregation actually converges - at least one missing ladder is
     backed by more than one role, and its headcount is exactly the sum of those
     roles' headcounts. Without this the view is just Phase 5 printed six times.
  3. ASSERT: the broad-SOC exclusion survives aggregation - no missing ladder is an
     occupation the org already has a role for (the team-level twin of
     validate_divergence.py CHECK 3).
  4. ASSERT: the top of the ladder is carried as unassessable WITH a reason, counts
     toward people_total, and contributes nothing to people_exposed. The regression
     guard against silently dropping roles the math cannot speak to.
  5. ASSERT: the totals are coherent - 0 <= exposed <= total, and no single missing
     ladder claims more people than the function has.
  6. ASSERT: repeated calls are byte-identical.

Exit code 0 = all assertions passed; non-zero = a check failed.

Run:  python tests/validate_team.py    (after the DB is built)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

# Make 'src' importable whether run as a script or a module.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.divergence import soc_base  # noqa: E402
from src.org import load_org  # noqa: E402
from src.team import format_team_drift, missing_ladders, role_drifts, team_drift  # noqa: E402
from src.vectors import DB_PATH  # noqa: E402

TOP_ROLE = "fin-director"           # the only unassessable role in the sample org
EXPECTED_PEOPLE = 18                # 4 + 6 + 3 + 3 + 1 + 1 from orgs/sample_finance.json

BAR = "=" * 70


def _verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def check_accounts_for_everyone(org, conn) -> bool:
    print(BAR)
    print("CHECK 1 - every role is carried and the people add up")
    print(BAR)

    td = team_drift(org, conn)
    ids = [d.role.role_id for d in td.roles]
    org_total = sum(r.headcount or 1 for r in org.roles)

    print(f"  roles in org ......... {len(org.roles)}")
    print(f"  roles in the view .... {len(td.roles)}  ({len(td.assessed)} assessed, "
          f"{len(td.unassessable)} not)")
    print(f"  people_total ......... {td.people_total}   (org sum {org_total})")
    print(f"  people_exposed ....... {td.people_exposed}")

    complete = sorted(ids) == sorted(r.role_id for r in org.roles)
    no_dupes = len(ids) == len(set(ids))
    adds_up = td.people_total == org_total == EXPECTED_PEOPLE

    passed = complete and no_dupes and adds_up
    print(f"\n  ASSERT  every role present ................... [{_verdict(complete)}]")
    print(f"  ASSERT  no role counted twice ................ [{_verdict(no_dupes)}]")
    print(f"  ASSERT  people_total == the org's headcount .. [{_verdict(adds_up)}]")
    return passed


def check_aggregation_converges(org, conn) -> bool:
    """
    The check that distinguishes Phase 6 from 'Phase 5, six times'.

    A missing ladder backed by one role is an anecdote. One backed by three roles
    covering twelve people is the org-design finding, so the headcount had better be
    exactly the sum over those roles - not a role count, not a double count.
    """
    print("\n" + BAR)
    print("CHECK 2 - the aggregation converges, and headcount sums correctly")
    print(BAR)

    td = team_drift(org, conn)
    shared = [m for m in td.missing if m.role_count > 1]

    print(f"  {'people':>6}  {'roles':>5}  {'cosine':>7}  {'margin':>8}  soc          title")
    print(f"  {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*12} {'-'*34}")
    for m in td.missing:
        print(f"  {m.headcount:>6}  {m.role_count:>5}  {m.best_similarity:>7.4f}  "
              f"{m.best_margin:>+8.4f}  {m.soc_code}  {m.title[:34]}")

    converges = bool(shared)
    sums_ok = all(
        m.headcount == sum(r.headcount or 1 for r in m.roles) for m in td.missing
    )
    ranked = [m.headcount for m in td.missing] == sorted(
        (m.headcount for m in td.missing), reverse=True
    )

    if shared:
        top = shared[0]
        print(f"\n  {top.title} ({top.soc_code}) is flagged by {top.role_count} roles:")
        for r in top.roles:
            print(f"    {r.title:<28} {r.headcount or 1:>2} people")
        print(f"    {'= headcount':<28} {top.headcount:>2}")

    passed = converges and sums_ok and ranked
    print(f"\n  ASSERT  a ladder is shared by 2+ roles ....... [{_verdict(converges)}]")
    print(f"  ASSERT  headcount == sum over those roles .... [{_verdict(sums_ok)}]")
    print(f"  ASSERT  ranked by people, most first ......... [{_verdict(ranked)}]")
    return passed


def check_org_occupations_excluded(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 3 - no 'missing' ladder is one the org already has")
    print(BAR)

    # Scan wide: an occupation the org has must not surface at any depth.
    drifts = role_drifts(org, conn, per_role_top_n=50)
    deep = missing_ladders(drifts, top_n=100)
    org_bases = {soc_base(r.onet_soc) for r in org.roles if r.onet_soc}
    offenders = [m.soc_code for m in deep if soc_base(m.soc_code) in org_bases]

    print(f"  occupations the org already has: {', '.join(sorted(org_bases))}")
    print(f"  scanned {len(deep)} aggregated destinations")
    print(f"  offending codes present: {offenders or 'none'}")

    passed = not offenders
    print(f"\n  ASSERT  no org occupation reported missing ... [{_verdict(passed)}]")
    return passed


def check_top_of_ladder_carried(org, conn) -> bool:
    """
    Regression guard: the roles the math cannot assess must still be visible.

    fin-director has no senior role, so it has no baseline and Phase 5 has nothing to
    say about it. Dropping it would make the table stop adding up to the org, which
    is precisely the failure this whole view exists to avoid.
    """
    print("\n" + BAR)
    print("CHECK 4 - the top of the ladder is carried, with a reason")
    print(BAR)

    td = team_drift(org, conn)
    top = next((d for d in td.roles if d.role.role_id == TOP_ROLE), None)

    if top is None:
        print(f"  {TOP_ROLE} is MISSING from the view entirely")
        print(f"\n  ASSERT  top of the ladder is carried ......... [{_verdict(False)}]")
        return False

    print(f"  role ............... {top.role.title}")
    print(f"  assessable ......... {top.assessable}")
    print(f"  reason ............. {top.reason}")
    print(f"  headcount .......... {top.headcount}")
    print(f"  flagged ............ {top.flagged}")

    present = True
    not_assessable = not top.assessable
    has_reason = bool(top.reason)
    counted = top.headcount > 0 and top.headcount <= td.people_total
    not_exposed = not top.flagged

    passed = present and not_assessable and has_reason and counted and not_exposed
    print(f"\n  ASSERT  carried, not dropped ................. [{_verdict(present)}]")
    print(f"  ASSERT  marked unassessable .................. [{_verdict(not_assessable)}]")
    print(f"  ASSERT  carries an explanation ............... [{_verdict(has_reason)}]")
    print(f"  ASSERT  counts toward people_total ........... [{_verdict(counted)}]")
    print(f"  ASSERT  contributes nothing to exposure ...... [{_verdict(not_exposed)}]")
    return passed


def check_totals_coherent(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 5 - the totals are coherent")
    print(BAR)

    td = team_drift(org, conn)
    exposed_ok = 0 <= td.people_exposed <= td.people_total
    ladders_ok = all(m.headcount <= td.people_total for m in td.missing)
    margins_ok = all(m.best_margin > 0 for m in td.missing)

    print(f"  people_exposed / people_total .... {td.people_exposed} / {td.people_total}")
    print(f"  largest missing ladder ........... "
          f"{max((m.headcount for m in td.missing), default=0)} people")
    print(f"  every best_margin positive ....... {margins_ok}")

    passed = exposed_ok and ladders_ok and margins_ok
    print(f"\n  ASSERT  0 <= exposed <= total ................ [{_verdict(exposed_ok)}]")
    print(f"  ASSERT  no ladder claims more than the org ... [{_verdict(ladders_ok)}]")
    print(f"  ASSERT  every margin beats its own baseline .. [{_verdict(margins_ok)}]")
    return passed


def check_deterministic(org, conn) -> bool:
    print("\n" + BAR)
    print("CHECK 6 - repeated calls return identical results")
    print(BAR)

    def shape(td):
        return (
            td.people_total,
            td.people_exposed,
            [(m.soc_code, m.headcount, m.best_similarity, m.best_margin, m.mean_margin)
             for m in td.missing],
            [(d.role.role_id, d.assessable, d.top_margin) for d in td.roles],
        )

    same = shape(team_drift(org, conn)) == shape(team_drift(org, conn))
    text_same = (
        format_team_drift(team_drift(org, conn), conn)
        == format_team_drift(team_drift(org, conn), conn)
    )

    print(f"  two independent team_drift() calls agree: {same}")
    print(f"  the rendered callout is byte-identical:   {text_same}")

    passed = same and text_same
    print(f"\n  ASSERT  deterministic ........................ [{_verdict(passed)}]")
    return passed


def show_callout(org, conn) -> None:
    print("\n" + BAR)
    print("THE DEPARTMENT CALLOUT (Phase 6 'money shot')")
    print(BAR)
    print(format_team_drift(team_drift(org, conn), conn))


def main() -> int:
    if not Path(DB_PATH).exists():
        print(f"ERROR: database not found at {DB_PATH}", file=sys.stderr)
        print("Build it first:  python build.py", file=sys.stderr)
        return 2

    conn = sqlite3.connect(DB_PATH)
    try:
        org = load_org()
        results = [
            ("accounts for everyone", check_accounts_for_everyone(org, conn)),
            ("aggregation converges", check_aggregation_converges(org, conn)),
            ("org occupations excluded", check_org_occupations_excluded(org, conn)),
            ("top of ladder carried", check_top_of_ladder_carried(org, conn)),
            ("totals coherent", check_totals_coherent(org, conn)),
            ("determinism", check_deterministic(org, conn)),
        ]
        show_callout(org, conn)
    finally:
        conn.close()

    print("\n" + BAR)
    failed = [name for name, ok in results if not ok]
    if not failed:
        print(f"RESULT: all {len(results)} Phase 6 assertions PASSED.")
        print(BAR)
        return 0
    print(f"RESULT: {len(failed)} assertion(s) FAILED: {', '.join(failed)}")
    print(BAR)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
