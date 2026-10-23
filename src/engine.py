"""
SkillPath Phase 3 - the skill-adjacency engine (pure, deterministic, NO LLM).

Ranks more-senior roles by how small a skill jump they are, walks the resulting
"skill-natural" ladder, and attaches the descriptors to develop at each step.

Public API
----------
    rank_candidates(org, role_id, conn, pool=None) -> list[Candidate]
    rank_socs(soc, candidate_socs, conn)           -> list[tuple[str, float]]
    next_role(org, role_id, conn)                  -> Candidate | None
    build_ladder(org, role_id, conn, max_steps=4)  -> Ladder

The LLM never touches any of this. Claude's job (Phase 2) is title -> SOC
matching and narrating the output; the numbers below are all O*NET-derived.

THE SAME-OCCUPATION CASE
------------------------
Two rungs of an org ladder often map to one O*NET occupation - Financial Analyst
and Senior Financial Analyst are both "Financial Quantitative Analysts", and that
is the *correct* mapping, not a data-entry slip. Identical SOC means an identical
vector, so cosine is exactly 1.00 and the element-wise gap is empty.

Rather than fudge the vector by org level (which would stop the adjacency being
O*NET-derived), such a step is reported honestly as same_occupation=True and its
recommendations are sourced from the next rung up that has a DIFFERENT SOC. The
step records where they came from in `recommendations_from`, so a substituted
recommendation is always visible rather than silent.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.org import Org, Role, next_level_roles, senior_roles
from src.vectors import is_rated, occupation_title, similarity, skill_gap_split

SAME_OCCUPATION_NOTE = (
    "same O*NET occupation - a depth/scope progression, so O*NET reports no skill delta"
)

# Diagram palette. ONE set for both themes - see ladder_dot's docstring for why.
#
# FONT must be a name Graphviz has metrics for, NOT a CSS stack: layout sizes each
# node from those metrics, while the browser renders the resolved font. Give it a
# stack it cannot measure and it lays out boxes too small, so the label spills past
# the rounded rect. "Helvetica" is measured correctly and resolves everywhere.
FONT = "Helvetica"
INK = "#0b0b0b"          # node text, on a light node fill
MUTED = "#898781"        # edges + labels: mode-invariant, >=3:1 on BOTH surfaces
MUTED_DARK = "#52514e"   # the node's second line, on a light node fill
START_FILL = "#cde2fb"   # "you are here"
START_LINE = "#2a78d6"
RUNG_FILL = "#f0efec"
RUNG_LINE = "#c3c2b7"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Candidate:
    """A possible next role, scored by skill adjacency."""

    role: Role
    similarity: float
    same_occupation: bool


@dataclass
class Step:
    """One rung-to-rung move on the skill-natural ladder."""

    from_role: Role
    to_role: Role
    similarity: float
    same_occupation: bool
    recommendations: dict[str, list[dict]]
    recommendations_from: str | None   # role_id the recommendations were derived against
    note: str | None = None

    @property
    def substituted(self) -> bool:
        """True when recommendations came from a rung other than `to_role`."""
        return self.recommendations_from not in (None, self.to_role.role_id)


@dataclass
class Ladder:
    """The full skill-natural progression out of one starting role."""

    start: Role
    steps: list[Step] = field(default_factory=list)

    @property
    def rungs(self) -> list[Role]:
        return [self.start] + [s.to_role for s in self.steps]

    @property
    def top(self) -> Role:
        return self.rungs[-1]


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

def _rateable(roles: list[Role], conn: sqlite3.Connection | None) -> list[Role]:
    """Drop roles whose SOC has no vector - similarity() would raise on them."""
    return [r for r in roles if r.onet_soc and is_rated(r.onet_soc, conn)]


def rank_candidates(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
    pool: list[Role] | None = None,
) -> list[Candidate]:
    """
    Score candidate next roles by cosine similarity, highest (= smallest jump) first.

    The default pool is every more-senior role in the org. `pool` overrides it, so
    a caller can rank against any role set - which is how Phase 5's divergence
    check will feed in occupations the org has no ladder for. Ties break on
    role_id for deterministic output.
    """
    role = org.get(role_id)
    if not role.onet_soc or not is_rated(role.onet_soc, conn):
        raise ValueError(
            f"Role '{role_id}' has no rated O*NET code (onet_soc={role.onet_soc!r}); "
            "cannot compute adjacency. Run validate_org first."
        )

    candidates = pool if pool is not None else senior_roles(org, role)
    scored = [
        Candidate(
            role=c,
            similarity=similarity(role.onet_soc, c.onet_soc, conn),
            same_occupation=(c.onet_soc == role.onet_soc),
        )
        for c in _rateable(candidates, conn)
        if c.role_id != role.role_id
    ]
    scored.sort(key=lambda c: (-c.similarity, c.role.role_id))
    return scored


def rank_socs(
    soc: str,
    candidate_socs: list[str],
    conn: sqlite3.Connection | None = None,
) -> list[tuple[str, float]]:
    """
    Rank raw SOC codes against one SOC, highest similarity first.

    The org-free counterpart to rank_candidates: Phase 5 scans O*NET occupations
    that are not in the org at all, so it has codes rather than Roles. Unrated
    codes are skipped rather than raising.
    """
    scored = [
        (c, similarity(soc, c, conn))
        for c in candidate_socs
        if c != soc and is_rated(c, conn)
    ]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored


def next_role(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
) -> Candidate | None:
    """
    The single natural next role: the smallest skill jump at the nearest level above.

    Returns None at the top of the ladder. Restricted to the next occupied level
    rather than all senior roles so the ladder advances one rung at a time
    instead of leaping straight to the most-similar director.
    """
    role = org.get(role_id)
    candidates = rank_candidates(org, role_id, conn, pool=next_level_roles(org, role))
    return candidates[0] if candidates else None


# --------------------------------------------------------------------------- #
# The ladder
# --------------------------------------------------------------------------- #

def _walk_rungs(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None,
    max_steps: int,
) -> list[tuple[Role, float, bool]]:
    """
    Greedily walk one rung at a time to the top: [(role, sim_from_previous, same_soc)].

    Element 0 is the starting role, with similarity 1.0 against itself. The walk
    is resolved up front so step-building can look ahead for a distinct SOC.
    """
    start = org.get(role_id)
    rungs: list[tuple[Role, float, bool]] = [(start, 1.0, False)]
    seen = {start.role_id}

    for _ in range(max_steps):
        nxt = next_role(org, rungs[-1][0].role_id, conn)
        if nxt is None or nxt.role.role_id in seen:
            break
        rungs.append((nxt.role, nxt.similarity, nxt.same_occupation))
        seen.add(nxt.role.role_id)

    return rungs


def build_ladder(
    org: Org,
    role_id: str,
    conn: sqlite3.Connection | None = None,
    max_steps: int = 4,
    top_n: int = 3,
) -> Ladder:
    """
    The skill-natural ladder out of `role_id`, with development recommendations.

    Each step carries the cosine similarity to the next rung and the top skills
    and knowledge areas to develop for it. Steps where both rungs share one SOC
    are flagged and borrow their recommendations from the next distinct-SOC rung
    (see the module docstring).
    """
    rungs = _walk_rungs(org, role_id, conn, max_steps)
    ladder = Ladder(start=rungs[0][0])

    for i in range(1, len(rungs)):
        from_role = rungs[i - 1][0]
        to_role, sim, same_soc = rungs[i]

        if not same_soc:
            ladder.steps.append(
                Step(
                    from_role=from_role,
                    to_role=to_role,
                    similarity=sim,
                    same_occupation=False,
                    recommendations=skill_gap_split(
                        from_role.onet_soc, to_role.onet_soc, conn, top_n=top_n
                    ),
                    recommendations_from=to_role.role_id,
                )
            )
            continue

        # Same occupation: look further up for a rung with a different SOC.
        source = next(
            (r for r, _sim, _same in rungs[i + 1:] if r.onet_soc != from_role.onet_soc),
            None,
        )
        if source is None:
            recommendations: dict[str, list[dict]] = {"skills": [], "knowledge": []}
            source_id = None
            note = (
                f"{SAME_OCCUPATION_NOTE}, and no higher rung maps to a different "
                "occupation, so there is nothing to derive recommendations from"
            )
        else:
            recommendations = skill_gap_split(
                from_role.onet_soc, source.onet_soc, conn, top_n=top_n
            )
            source_id = source.role_id
            note = (
                f"{SAME_OCCUPATION_NOTE}; recommendations below are the run-up to "
                f"{source.title}, the next rung that maps to a different occupation"
            )

        ladder.steps.append(
            Step(
                from_role=from_role,
                to_role=to_role,
                similarity=sim,
                same_occupation=True,
                recommendations=recommendations,
                recommendations_from=source_id,
                note=note,
            )
        )

    return ladder


# --------------------------------------------------------------------------- #
# Rendering (plain text, spec section 7 style)
# --------------------------------------------------------------------------- #

def format_ladder(ladder: Ladder, conn: sqlite3.Connection | None = None) -> str:
    """Render a Ladder the way the spec's section 7 worked example lays it out."""
    lines: list[str] = []
    start = ladder.start
    lines.append(f"{start.title}  ({start.onet_soc} {occupation_title(start.onet_soc, conn)})")

    for step in ladder.steps:
        marker = "  same occupation" if step.same_occupation else ""
        lines.append(f"      |  cosine {step.similarity:.3f}{marker}")
        lines.append("      v")
        to = step.to_role
        lines.append(f"{to.title}  ({to.onet_soc} {occupation_title(to.onet_soc, conn)})")
        if step.note:
            lines.append(f"   note: {step.note}")

        recs = step.recommendations
        if not recs["skills"] and not recs["knowledge"]:
            lines.append("   -> no development recommendations available")
            continue
        if recs["skills"]:
            names = ", ".join(f"{g['element_name']} (+{g['delta']:.2f})" for g in recs["skills"])
            lines.append(f"   -> develop (skills):    {names}")
        if recs["knowledge"]:
            names = ", ".join(f"{g['element_name']} (+{g['delta']:.2f})" for g in recs["knowledge"])
            lines.append(f"   -> develop (knowledge): {names}")

    return "\n".join(lines)


def ladder_dot(
    ladder: Ladder,
    conn: sqlite3.Connection | None = None,
    size: str | None = None,
) -> str:
    """
    Render a Ladder as Graphviz DOT, for st.graphviz_chart.

    Emits DOT source rather than using the `graphviz` package, because Streamlit
    renders DOT client-side - so the app needs no Graphviz system binary installed.

    `size` is an optional Graphviz bounding box in inches, e.g. "7,6". It is emitted
    with the "!" suffix so the graph is scaled to fill it rather than only shrunk to
    fit. The demo view (spec section 10) needs the whole ladder inside one viewport
    with no scrolling; the default None leaves the graph at its natural size.

    The background is transparent, so the SVG sits on whichever theme the viewer
    has, and DOT has no media queries - so ONE palette has to read on both. Nodes
    are light fills with near-black ink, legible either way; edges and their labels
    use #898781, the one muted ink that is mode-invariant and clears 3:1 against
    both surfaces.

    The starting rung carries the accent tint - "you are here". Every rung above it
    is neutral: the rungs are a path, not a magnitude, so ramping their colour would
    encode something the data does not say. The cosine on each edge carries distance.

    Labels are HTML-like so the role title can outweigh its O*NET code, which drops
    to a smaller muted second line instead of a same-size continuation.
    """

    def esc(text: str) -> str:
        """Escape for an HTML-like DOT label: & first, then the angle brackets."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    lines = [
        "digraph ladder {",
        "  rankdir=TB;",
        '  bgcolor="transparent";',
    ]
    if size:
        # Streamlit scales the SVG to the container width, so height follows from the
        # aspect ratio: the only real lever on how tall this renders is the gap between
        # ranks. Tighten it (and the bounding box) so a five-rung ladder still fits one
        # viewport without scrolling.
        lines.append(f'  size="{size}";')
        lines.append('  ranksep="0.22";')
        lines.append('  nodesep="0.12";')
    lines += [
        f'  node [shape=box, style="rounded,filled", fontname="{FONT}",'
        f' fontsize=11, margin="0.22,0.14", penwidth=1.1,'
        f' color="{RUNG_LINE}", fillcolor="{RUNG_FILL}", fontcolor="{INK}"];',
        f'  edge [fontname="{FONT}", fontsize=10, penwidth=1.1,'
        f' color="{MUTED}", fontcolor="{MUTED}", arrowsize=0.7];',
    ]

    for i, role in enumerate(ladder.rungs):
        occ = occupation_title(role.onet_soc, conn) if role.onet_soc else "unmatched"
        label = (
            f"<<B>{esc(role.title)}</B>"
            f'<BR/><FONT POINT-SIZE="9" COLOR="{MUTED_DARK}">'
            f'{esc(role.onet_soc or "")} {esc(occ)}</FONT>>'
        )
        if i == 0:
            lines.append(
                f"  r{i} [label={label}, fillcolor=\"{START_FILL}\","
                f' color="{START_LINE}", penwidth=1.6];'
            )
        else:
            lines.append(f"  r{i} [label={label}];")

    for i, step in enumerate(ladder.steps):
        label = f"cosine {step.similarity:.2f}"
        style = ""
        if step.same_occupation:
            label += "  ·  same occupation"
            style = ", style=dashed"
        lines.append(f'  r{i} -> r{i + 1} [label="  {label}"{style}];')

    lines.append("}")
    return "\n".join(lines)

