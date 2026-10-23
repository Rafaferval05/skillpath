"""
SkillPath Phase 6 - the team-level drift view (pure, deterministic, NO LLM).

Phase 5 answers the question for ONE role: "this analyst's skills point at
Actuaries, a track you have no ladder for." That is a single data point. The
consulting question is the department one - where is this whole function's skill
base drifting, and how many PEOPLE are behind that drift?

This module runs the Phase 5 divergence across every role in the org and groups
the results by destination occupation, weighted by headcount. It computes no new
similarity math whatsoever: every number here comes out of src/divergence.py,
which in turn comes out of src/vectors.py.

Public API
----------
    role_drifts(org, conn, per_role_top_n=10) -> list[RoleDrift]
    missing_ladders(drifts, top_n=5)          -> list[MissingLadder]
    team_drift(org, conn, top_n=5)            -> TeamDrift
    format_team_drift(td, conn)               -> str

WHY HEADCOUNT IS THE AGGREGATING UNIT, NOT MARGIN
-------------------------------------------------
Each role's off-ladder margin is measured against ITS OWN baseline - the best step
up that particular role is offered (see _best_senior in divergence.py). The clerk's
+0.06 and the analyst's +0.14 are therefore fractions of different denominators, and
summing or averaging them across roles quietly mixes scales.

Headcount does not have that problem: six analysts are six people no matter which
baseline their role was scored against. So headcount is the number this view leads
with, and margins are reported per-role and per-destination but never rolled up into
one org-wide "drift index". Inventing a composite score is exactly what this codebase
refused to do when it rejected the spec's absolute 0.85 cosine threshold, and the
same refusal applies here.

MissingLadder.mean_margin IS a weighted mean, but only across the roles pointing at
that ONE occupation - it answers "how much closer is this destination, typically, for
the people drifting toward it", which is a question about a single destination rather
than a summary of the whole function.

WHY UNASSESSABLE ROLES ARE CARRIED, NOT DROPPED
-----------------------------------------------
divergence() raises on a role with no rated SOC, and has nothing to say at the top of
the ladder (no senior roles => no baseline => the comparison has no meaning). The
Finance Director is exactly that case. Silently dropping such roles would produce a
table that does not add up to the org - "6 roles, 5 assessed, 1 at the top of the
ladder" is the honest rendering, so every role gets a RoleDrift carrying an
`assessable` flag and a `reason`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from src.divergence import Divergence, divergence
from src.org import Org, Role
from src.vectors import is_rated

# Roles are scanned deeper than the team view displays: convergence is the entire
# point, and a destination sitting 6th for three separate roles is a bigger finding
# than one sitting 1st for a single role. Truncating per-role at the display width
# would hide exactly that.
PER_ROLE_TOP_N = 10

# A role with no headcount counts as one person, never zero - zero would silently
# erase the role from the department total. The RoleDrift records the assumption so
# the render can footnote it.
ASSUMED_HEADCOUNT = 1

NO_CODE = "no rated O*NET code - cannot build a vector"
TOP_OF_LADDER = "top of the ladder - no senior role to compare against"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RoleDrift:
    """One role's Phase 5 finding, plus the headcount behind it."""

    role: Role
    headcount: int
    headcount_assumed: bool
    assessable: bool
    reason: str | None                  # why not, when assessable is False
    divergence: Divergence | None

    @property
    def flagged(self) -> bool:
        return self.divergence is not None and self.divergence.flagged

    @property
    def top_margin(self) -> float:
        """Margin of this role's closest off-ladder occupation; 0.0 when none."""
        if not self.divergence or not self.divergence.off_ladder:
            return 0.0
        return self.divergence.off_ladder[0].margin

    @property
    def path_disagrees(self) -> bool:
        return self.divergence is not None and not self.divergence.path.agrees


@dataclass(frozen=True)
class MissingLadder:
    """
    An occupation several roles' skills point at, and nobody can be promoted into.

    `roles` is every org role that flagged it; `headcount` is the people in those
    roles. That pair is the finding: not "an occupation scored well" but "this many
    people sit in roles whose skills point here, and you have no ladder for it".
    """

    soc_code: str
    title: str
    roles: list[Role] = field(compare=False)
    headcount: int = 0
    best_similarity: float = 0.0
    best_margin: float = 0.0
    mean_margin: float = 0.0            # headcount-weighted, across `roles` only

    @property
    def role_count(self) -> int:
        return len(self.roles)

    def __str__(self) -> str:
        return (
            f"{self.title} ({self.soc_code}) - {self.headcount} people "
            f"across {self.role_count} role(s)"
        )


@dataclass(frozen=True)
class TeamDrift:
    """The Phase 6 finding for a whole function."""

    org: Org
    roles: list[RoleDrift]
    missing: list[MissingLadder]
    people_total: int
    people_exposed: int
    path_disagreements: list[RoleDrift]

    @property
    def assessed(self) -> list[RoleDrift]:
        return [d for d in self.roles if d.assessable]

    @property
    def unassessable(self) -> list[RoleDrift]:
        return [d for d in self.roles if not d.assessable]

    @property
    def headcount_assumed(self) -> bool:
        """True when any role had no headcount and was counted as one person."""
        return any(d.headcount_assumed for d in self.roles)

    @property
    def flagged(self) -> bool:
        return bool(self.missing) or bool(self.path_disagreements)


# --------------------------------------------------------------------------- #
# 1) Per-role drift
# --------------------------------------------------------------------------- #

def _headcount(role: Role) -> tuple[int, bool]:
    """(effective headcount, whether it was assumed). See ASSUMED_HEADCOUNT."""
    if role.headcount is None:
        return ASSUMED_HEADCOUNT, True
    return int(role.headcount), False


def role_drifts(
    org: Org,
    conn: sqlite3.Connection | None = None,
    per_role_top_n: int = PER_ROLE_TOP_N,
) -> list[RoleDrift]:
    """
    Run the Phase 5 divergence for every role, in the org's own level order.

    Preconditions are checked the way divergence._require_rated checks them rather
    than by catching exceptions, so a genuine bug still raises instead of being
    silently recorded as "unassessable".
    """
    drifts: list[RoleDrift] = []

    for role in sorted(org.roles, key=lambda r: (r.level, r.role_id)):
        headcount, assumed = _headcount(role)

        if not role.onet_soc or not is_rated(role.onet_soc, conn):
            drifts.append(RoleDrift(role, headcount, assumed, False, NO_CODE, None))
            continue

        div = divergence(org, role.role_id, conn, top_n=per_role_top_n)

        # No baseline means no senior role to compare against - the top of the
        # ladder. divergence() returns an empty, unflagged result there; reporting
        # it as an ordinary "no divergence" would read as a clean bill of health.
        if div.baseline_role is None:
            drifts.append(RoleDrift(role, headcount, assumed, False, TOP_OF_LADDER, div))
            continue

        drifts.append(RoleDrift(role, headcount, assumed, True, None, div))

    return drifts


# --------------------------------------------------------------------------- #
# 2) Aggregation: the ladders this function is missing
# --------------------------------------------------------------------------- #

def missing_ladders(drifts: list[RoleDrift], top_n: int = 5) -> list[MissingLadder]:
    """
    Group every role's off-ladder matches by destination occupation.

    Sorted by headcount first - the department's question is "how many people" -
    then by best margin, then by SOC code so repeated runs are byte-identical.

    The broad-SOC exclusion (an occupation the org already has a role for is not a
    missing ladder) is applied inside off_ladder_matches; this layer inherits it and
    must not re-implement it.
    """
    grouped: dict[str, dict] = {}

    for d in drifts:
        if not d.assessable or d.divergence is None:
            continue
        for m in d.divergence.off_ladder:
            g = grouped.setdefault(
                m.soc_code,
                {
                    "title": m.title,
                    "roles": [],
                    "headcount": 0,
                    "best_similarity": 0.0,
                    "best_margin": 0.0,
                    "weighted": 0.0,
                },
            )
            g["roles"].append(d.role)
            g["headcount"] += d.headcount
            g["best_similarity"] = max(g["best_similarity"], m.similarity)
            g["best_margin"] = max(g["best_margin"], m.margin)
            g["weighted"] += m.margin * d.headcount

    ladders = [
        MissingLadder(
            soc_code=soc,
            title=g["title"],
            roles=g["roles"],
            headcount=g["headcount"],
            best_similarity=round(g["best_similarity"], 4),
            best_margin=round(g["best_margin"], 4),
            mean_margin=round(g["weighted"] / g["headcount"], 4) if g["headcount"] else 0.0,
        )
        for soc, g in grouped.items()
    ]
    ladders.sort(key=lambda m: (-m.headcount, -m.best_margin, m.soc_code))
    return ladders[:top_n]


# --------------------------------------------------------------------------- #
# 3) The combined finding
# --------------------------------------------------------------------------- #

def team_drift(
    org: Org,
    conn: sqlite3.Connection | None = None,
    top_n: int = 5,
    per_role_top_n: int = PER_ROLE_TOP_N,
) -> TeamDrift:
    """The full Phase 6 report: per-role drift, aggregated into missing ladders."""
    drifts = role_drifts(org, conn, per_role_top_n=per_role_top_n)

    return TeamDrift(
        org=org,
        roles=drifts,
        missing=missing_ladders(drifts, top_n=top_n),
        # people_total counts EVERY role, assessable or not: the function has those
        # people whether or not the math can say anything about them.
        people_total=sum(d.headcount for d in drifts),
        people_exposed=sum(d.headcount for d in drifts if d.flagged),
        path_disagreements=[d for d in drifts if d.path_disagrees],
    )


# --------------------------------------------------------------------------- #
# Rendering (plain text, spec section 7 style)
# --------------------------------------------------------------------------- #

def _people(n: int) -> str:
    return f"{n} person" if n == 1 else f"{n} people"


def format_team_drift(td: TeamDrift, conn: sqlite3.Connection | None = None) -> str:
    """Render a TeamDrift as a department-level callout."""
    lines: list[str] = []
    org = td.org

    lines.append(f"[!] Function-level skill drift - {org.company}, {org.function}")
    lines.append("")
    lines.append(
        f"  {len(td.roles)} roles, {_people(td.people_total)}. "
        f"{len(td.assessed)} role(s) assessed, {len(td.unassessable)} not."
    )

    if not td.flagged:
        lines.append("")
        lines.append(
            "  No drift. Every assessed role's best step up is already the smallest "
            "skill jump available, and no occupation outside the org is closer. This "
            "function's published ladders match where its skills point."
        )
        return "\n".join(lines)

    if td.missing:
        top = td.missing[0]
        lines.append("")
        lines.append(
            f"  {top.headcount} of {td.people_total} people sit in roles whose skills "
            f"point at {top.title} ({top.soc_code}) - an occupation this function has "
            f"NO ladder for. It is an off-ladder match for {top.role_count} role(s): "
            + ", ".join(r.title for r in top.roles)
            + "."
        )
        lines.append("")
        lines.append("  Ladders this function is missing, by people behind them:")
        lines.append(
            f"    {'people':>6}  {'roles':>5}  {'cosine':>7}  {'margin':>8}  "
            "soc          title"
        )
        lines.append(f"    {'-'*6}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*12} {'-'*36}")
        for m in td.missing:
            lines.append(
                f"    {m.headcount:>6}  {m.role_count:>5}  {m.best_similarity:>7.4f}  "
                f"{m.best_margin:>+8.4f}  {m.soc_code}  {m.title[:36]}"
            )
        lines.append("")
        lines.append(
            f"  Read: {td.people_exposed} of {td.people_total} people are in roles whose "
            "skills point somewhere this function cannot promote them - retention risk "
            "concentrated in one missing track, not scattered across individuals."
        )

    if td.path_disagreements:
        lines.append("")
        lines.append("  Roles whose org chart and skill ladder disagree:")
        for d in td.path_disagreements:
            p = d.divergence.path
            lines.append(
                f"    {d.role.title} ({_people(d.headcount)}): chart says "
                f"{p.org_next.title} ({p.org_next_similarity:.2f}), skills say "
                f"{p.skill_next.title} ({p.skill_next_similarity:.2f}), "
                f"{p.margin:+.2f} closer."
            )

    if td.unassessable:
        lines.append("")
        lines.append("  Not assessed:")
        for d in td.unassessable:
            lines.append(f"    {d.role.title} ({_people(d.headcount)}) - {d.reason}")

    lines.append("")
    lines.append(
        "  Note: each role's margin is measured against its OWN best step up, so "
        "margins are comparable within a destination but not summed across the "
        "function. Headcount is the unit that aggregates cleanly."
    )
    if td.headcount_assumed:
        lines.append(
            f"  Note: role(s) with no headcount were counted as "
            f"{ASSUMED_HEADCOUNT} person each."
        )

    return "\n".join(lines)
