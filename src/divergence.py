"""
SkillPath Phase 5 - the org-vs-skill divergence flag (pure, deterministic, NO LLM).

The differentiator from spec section 7. Phases 0-4 show the ladder that already
exists; this module surfaces the ladder that *should* exist, in two forms:

    1. OFF-LADDER OPPORTUNITY - O*NET occupations whose skill profile sits closer
       to this role than anything the org actually offers as a step up. "Your
       analysts' skills point somewhere you don't offer a path" -> retention risk,
       or an opportunity to build a lateral track.

    2. PATH DIVERGENCE - the role the org chart says is next (follow reports_to)
       versus the senior role that is genuinely the smallest skill jump. When
       those disagree, the published ladder is not the skill-natural one.

Public API
----------
    off_ladder_matches(org, role_id, conn, top_n=5) -> list[OffLadderMatch]
    path_divergence(org, role_id, conn)             -> PathDivergence
    divergence(org, role_id, conn, top_n=5)         -> Divergence
    format_divergence(div, conn)                    -> str

WHY THE BASELINE IS RELATIVE, NOT AN ABSOLUTE COSINE
----------------------------------------------------
The spec's worked example flags a match at "cosine 0.85". Taken as an absolute
threshold that is meaningless here: because every vector entry is a non-negative
LV x IM product, cosine over real occupations is compressed into a narrow high
band. Against Financial Quantitative Analysts the median of all 894 rated
occupations is 0.74 and the 95th percentile is 0.86 - so "0.85" would flag a
twentieth of the entire economy, Cooks (0.74) very nearly included.

What actually carries meaning is the comparison the org itself sets: is this
occupation a *smaller skill jump than the best step up the org offers*? That
baseline is self-calibrating, needs no magic number, and states the finding in
the only terms an HR reader cares about - better than what we already have.

WHY THERE IS NO JOB-ZONE / SENIORITY FILTER
-------------------------------------------
O*NET's Job Zone (preparation required, 1-5) looks like a seniority proxy and is
not one. Filtering to "zone >= the current role's" drops Actuaries (the single
closest match at 0.976), Business Intelligence Analysts and Credit Analysts,
while keeping Astronomers, Physicists and Postsecondary Teachers - all zone 5 and
all absurd as finance-career moves. Ranking by skill adjacency and taking the top
few does the job the zone filter was supposed to do, and does it better.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

from src.org import Org, Role, org_path, senior_roles
from src.vectors import is_rated, occupation_matrix, occupation_title

# An occupation is "already in the org" when it shares a BROAD SOC with any org
# role: 11-3031.01 Treasurers and 11-3031.03 Investment Fund Managers are detail
# codes under 11-3031.00 Financial Managers, so an org with Financial Managers
# demonstrably has that ladder and should not be told it is missing.
SOC_BASE_LEN = 7   # '11-3031.00' -> '11-3031'


def soc_base(soc: str) -> str:
    return soc[:SOC_BASE_LEN]


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OffLadderMatch:
    """An O*NET occupation closer to this role than anything the org offers."""

    soc_code: str
    title: str
    similarity: float
    margin: float   # similarity - the org's best step up; > 0 by construction

    def __str__(self) -> str:
        return f"{self.title} ({self.soc_code}) cosine {self.similarity:.2f}"


@dataclass(frozen=True)
class PathDivergence:
    """Where the org chart says you go next, versus where your skills point."""

    role: Role
    org_next: Role | None            # follow reports_to
    org_next_similarity: float | None
    skill_next: Role | None          # smallest skill jump among senior roles
    skill_next_similarity: float | None

    @property
    def comparable(self) -> bool:
        """False at the top of the ladder, or when either side has no vector."""
        return self.org_next is not None and self.skill_next is not None

    @property
    def agrees(self) -> bool:
        """True when the org's next rung IS the smallest skill jump."""
        if not self.comparable:
            return True
        return self.org_next.role_id == self.skill_next.role_id

    @property
    def margin(self) -> float:
        """How much closer the skill-natural rung is than the org's own next rung."""
        if not self.comparable or self.agrees:
            return 0.0
        return self.skill_next_similarity - self.org_next_similarity


@dataclass(frozen=True)
class Divergence:
    """The full Phase 5 finding for one role."""

    role: Role
    baseline_role: Role | None       # best step up the org actually offers
    baseline_similarity: float | None
    off_ladder: list[OffLadderMatch]
    path: PathDivergence

    @property
    def flagged(self) -> bool:
        """True when there is anything worth showing the user."""
        return bool(self.off_ladder) or not self.path.agrees


# --------------------------------------------------------------------------- #
# The baseline: the best step up the org actually offers
# --------------------------------------------------------------------------- #

def _distinct_senior(org: Org, role: Role, conn: sqlite3.Connection | None) -> list[Role]:
    """
    Senior roles that are a genuine skill move - rated, and a DIFFERENT occupation.

    Same-SOC rungs (Financial Analyst -> Senior Financial Analyst) score exactly
    1.0, so leaving them in would make the baseline unbeatable and silence the
    flag entirely. A depth/scope promotion within one occupation is not the step
    up this comparison is about.
    """
    return [
        r
        for r in senior_roles(org, role)
        if r.onet_soc and r.onet_soc != role.onet_soc and is_rated(r.onet_soc, conn)
    ]


def _require_rated(org: Org, role_id: str, conn: sqlite3.Connection | None) -> Role:
    role = org.get(role_id)
    if not role.onet_soc or not is_rated(role.onet_soc, conn):
        raise ValueError(
            f"Role '{role_id}' has no rated O*NET code (onet_soc={role.onet_soc!r}); "
            "cannot compute divergence. Run validate_org first."
        )
    return role


def _sims_for(role: Role, conn: sqlite3.Connection | None):
    """(similarity-vector, soc->row index) for one role against every occupation."""
    socs, unit = occupation_matrix(conn)
    index = {s: i for i, s in enumerate(socs)}
    return socs, unit @ unit[index[role.onet_soc]], index


def _best_senior(
    org: Org,
    role: Role,
    sims: np.ndarray,
    index: dict[str, int],
    conn: sqlite3.Connection | None,
) -> tuple[Role | None, float | None]:
    """
    The org's best step up: (role, similarity), or (None, None) at the top.

    Single source of truth for the baseline. off_ladder_matches() measures its
    margins against this number and divergence() reports it, so computing it
    twice would let the printed baseline drift out of step with the margins.
    Ties break on role_id ascending, matching rank_candidates() in engine.py.
    """
    senior = _distinct_senior(org, role, conn)
    if not senior:
        return None, None
    best = min(senior, key=lambda r: (-sims[index[r.onet_soc]], r.role_id))
    return best, round(float(sims[index[best.onet_soc]]), 4)


# --------------------------------------------------------------------------- #
# 1) Off-ladder opportunities
# --------------------------------------------------------------------------- #

def off_ladder_matches(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
    top_n: int = 5,
) -> list[OffLadderMatch]:
    """
    O*NET occupations closer to this role than the org's own best step up.

    Excludes any occupation sharing a broad SOC with a role the org already has,
    and the role's own occupation. Sorted by similarity, ties broken on SOC code
    so the output is deterministic. Empty when the org already offers the closest
    move available - which is the healthy case, not a failure.
    """
    role = _require_rated(org, role_id, conn)
    socs, sims, index = _sims_for(role, conn)

    _best, baseline = _best_senior(org, role, sims, index, conn)
    if baseline is None:
        return []   # nothing to compare against; the flag would have no meaning

    org_bases = {soc_base(r.onet_soc) for r in org.roles if r.onet_soc}
    org_bases.add(soc_base(role.onet_soc))

    matches = [
        OffLadderMatch(
            soc_code=soc,
            title=occupation_title(soc, conn),
            similarity=round(float(sims[i]), 4),
            margin=round(float(sims[i]) - baseline, 4),
        )
        for i, soc in enumerate(socs)
        if soc_base(soc) not in org_bases and float(sims[i]) > baseline
    ]
    matches.sort(key=lambda m: (-m.similarity, m.soc_code))
    return matches[:top_n]


# --------------------------------------------------------------------------- #
# 2) Org path vs skill path
# --------------------------------------------------------------------------- #

def path_divergence(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
) -> PathDivergence:
    """
    Compare the reports_to next rung with the smallest-skill-jump senior role.

    org_next comes from org_path() - the org chart's own answer. skill_next is the
    closest senior role by cosine, excluding same-occupation rungs for the reason
    given in _distinct_senior. Both similarities are measured against the starting
    role so the two numbers are directly comparable.
    """
    role = _require_rated(org, role_id, conn)
    socs, sims, index = _sims_for(role, conn)

    def sim_to(r: Role) -> float | None:
        if not r.onet_soc or r.onet_soc not in index:
            return None
        return round(float(sims[index[r.onet_soc]]), 4)

    chain = org_path(org, role_id)
    org_next = chain[1] if len(chain) > 1 else None
    skill_next, _sim = _best_senior(org, role, sims, index, conn)

    return PathDivergence(
        role=role,
        org_next=org_next,
        org_next_similarity=sim_to(org_next) if org_next else None,
        skill_next=skill_next,
        skill_next_similarity=sim_to(skill_next) if skill_next else None,
    )


# --------------------------------------------------------------------------- #
# 3) The combined finding
# --------------------------------------------------------------------------- #

def divergence(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
    top_n: int = 5,
) -> Divergence:
    """The full Phase 5 report for one role: off-ladder matches + path divergence."""
    role = _require_rated(org, role_id, conn)
    socs, sims, index = _sims_for(role, conn)
    baseline_role, baseline_similarity = _best_senior(org, role, sims, index, conn)

    return Divergence(
        role=role,
        baseline_role=baseline_role,
        baseline_similarity=baseline_similarity,
        off_ladder=off_ladder_matches(org, role_id, conn, top_n=top_n),
        path=path_divergence(org, role_id, conn),
    )


# --------------------------------------------------------------------------- #
# Rendering (plain text, spec section 7 style)
# --------------------------------------------------------------------------- #

def format_divergence(div: Divergence, conn: sqlite3.Connection | None = None) -> str:
    """Render a Divergence as the spec's section 7 callout."""
    role = div.role
    lines: list[str] = []

    if not div.flagged:
        lines.append("No divergence flagged.")
        lines.append(
            f"  The org's own next step up for {role.title} is also the smallest skill "
            "jump available, and no O*NET occupation outside the org is closer. The "
            "published ladder matches where these skills point."
        )
        return "\n".join(lines)

    lines.append(f"[!] Org-vs-skill divergence - {role.title} ({role.onet_soc})")
    lines.append("")

    if div.off_ladder:
        best = div.off_ladder[0]
        offered = (
            f"{div.baseline_role.title} at cosine {div.baseline_similarity:.2f}"
            if div.baseline_role
            else "nothing"
        )
        lines.append(
            f"  This role's skill vector sits closest to {best.title} "
            f"({best.soc_code}) at cosine {best.similarity:.2f} - a track your org "
            f"has NO ladder for. The best step up you do offer is {offered}, "
            f"{best.margin:+.2f} further away."
        )
        lines.append("")
        lines.append("  Closest occupations your org has no role for:")
        lines.append(f"    {'cosine':>7}  {'vs offered':>10}  soc          title")
        lines.append(f"    {'-' * 7}  {'-' * 10}  {'-' * 12} {'-' * 40}")
        for m in div.off_ladder:
            lines.append(
                f"    {m.similarity:>7.4f}  {m.margin:>+10.4f}  {m.soc_code}  {m.title[:40]}"
            )
        lines.append("")
        lines.append(
            "  Read: these people's skills point somewhere you cannot promote them "
            "-> retention risk, or an opportunity to build the lateral track."
        )

    path = div.path
    if not path.agrees:
        if div.off_ladder:
            lines.append("")
        lines.append(
            f"  Your org chart sends {role.title} to {path.org_next.title} "
            f"(cosine {path.org_next_similarity:.2f}), but the smallest skill jump is "
            f"{path.skill_next.title} (cosine {path.skill_next_similarity:.2f}), "
            f"{path.margin:+.2f} closer."
        )
        lines.append(
            "  Read: the published ladder is not the skill-natural one - the "
            "promotion you offer is a bigger jump than a move you don't."
        )

    return "\n".join(lines)
