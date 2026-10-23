"""
SkillPath Phases 4, 6 and 7 - the Streamlit front end.

    streamlit run app.py

Ties the three layers together in the order the spec's section 6 user flow lays out:
upload org -> see where the whole function's skills are drifting -> pick role ->
confirm the O*NET match -> answer clarifying questions -> compute -> render the
progression map -> export a one-page summary.

The function-level view (section 2) sits ABOVE the role picker and above the
st.stop() that guards an unmatched role, so the department finding still renders
for an org where some individual role cannot be matched yet.

DEMO VIEW (Phase 7, the spec's section 10 money shot)
-----------------------------------------------------
The sidebar's 'Demo view' toggle renders one screen - role picker, ladder, callout -
and then calls st.stop(). Sections 1-7 below it are simply never reached, which is
why they could be left untouched rather than indented into an `if`. The demo adds no
logic of its own: it re-renders build_ladder / divergence / team_drift results that
the full view already shows, minus the tables, expanders and export.

This file is presentation only. Every number on screen comes from src/engine.py and
src/vectors.py; everything Claude does is confined to src/matching.py.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from datetime import date
from pathlib import Path

import streamlit as st

from src.divergence import divergence
from src.engine import build_ladder, ladder_dot, rank_candidates
from src.org import SAMPLE_ORG_PATH, Org, errors, load_org, org_path, validate_org
from src.team import team_drift
from src.vectors import DB_PATH, is_rated, occupation_title

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="SkillPath", page_icon="🪜", layout="wide")


# --------------------------------------------------------------------------- #
# Design tokens
# --------------------------------------------------------------------------- #
#
# The palette lives in .streamlit/config.toml; these are the few values Streamlit
# has no theme setting for - the callout cards and the ladder diagram.
#
# Light and dark are SELECTED sets, not one set auto-flipped: each is stepped for
# its own surface (#fcfcfb / #1a1a19) and checked against it.
#
# Colour does one job each. Blue is identity and interaction only. Orange means
# "a gap we found" and is sub-3:1 on the light surface by design, so it is never
# the only signal - every card pairs it with an icon and a heading, and the light
# heading ink is a darker step of the same hue. Green is reserved for the
# genuinely healthy case, red for validation errors. A missing ladder is an
# opportunity, not an error, so it is never red.

TOKENS = {
    "light": {
        "surface": "#fcfcfb",
        "muted": "#52514e",
        "gap_tint": "#fdf1ea", "gap_rule": "#ec835a", "gap_ink": "#c2410c",
        "ok_tint": "#eef8ee", "ok_rule": "#0ca30c", "ok_ink": "#006300",
        "info_tint": "#eaf2fd", "info_rule": "#2a78d6", "info_ink": "#1c5cab",
    },
    "dark": {
        "surface": "#1a1a19",
        "muted": "#c3c2b7",
        "gap_tint": "#2a1d16", "gap_rule": "#ec835a", "gap_ink": "#ec835a",
        "ok_tint": "#13251a", "ok_rule": "#0ca30c", "ok_ink": "#0ca30c",
        "info_tint": "#16202c", "info_rule": "#3987e5", "info_ink": "#3987e5",
    },
}

# Graphviz has no media queries, so the diagram gets ONE palette that has to read
# on both surfaces. #898781 (the palette's muted ink) is mode-invariant and clears
# 3:1 both ways, which is why edges and their labels use it; nodes are light fills
# with near-black ink, legible against either background.
DOT_INK = "#0b0b0b"
DOT_MUTED = "#898781"
DOT_START_FILL = "#cde2fb"     # "you are here" - the accent, tinted
DOT_START_LINE = "#2a78d6"
DOT_RUNG_FILL = "#f0efec"
DOT_RUNG_LINE = "#c3c2b7"


def theme_mode() -> str:
    """The viewer's ACTIVE theme, not the configured default - they can differ."""
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:  # noqa: BLE001 - older Streamlit, or no browser context yet
        return "dark"


T = TOKENS[theme_mode()]


def callout(kind: str, icon: str, heading: str, body: str) -> None:
    """
    One card: a tinted panel with a coloured left rule, an icon and a heading.

    Replaces st.warning/st.error for findings. Streamlit's own alerts carry a
    fixed red/amber that reads as "something broke"; a role whose skills point
    off-ladder is a finding, not a fault, so it gets its own restrained treatment
    and red stays reserved for real errors.
    """
    tint, rule, ink = T[f"{kind}_tint"], T[f"{kind}_rule"], T[f"{kind}_ink"]
    st.markdown(
        f"""<div style="background:{tint};border-left:3px solid {rule};
             border-radius:0.5rem;padding:1rem 1.15rem;margin:0 0 0.85rem 0;">
          <div style="color:{ink};font-weight:650;font-size:1.02rem;
               line-height:1.35;margin-bottom:0.4rem;">{icon} {heading}</div>
          <div style="font-size:0.94rem;line-height:1.5;">{body}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #

@st.cache_resource
def get_conn() -> sqlite3.Connection:
    """One shared read-only connection. check_same_thread=False for Streamlit's threads."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)


@st.cache_resource
def get_claude():
    from src.matching import get_client

    return get_client()


def load_org_from_upload(upload) -> Org:
    """Persist an uploaded file to a temp path so load_org's validation path is reused."""
    tmp = PROJECT_ROOT / ".uploaded_org.json"
    tmp.write_bytes(upload.getvalue())
    return load_org(tmp)


def current_org(base: Org) -> Org:
    """Apply any user-confirmed SOC overrides on top of the loaded org."""
    overrides = st.session_state.get("soc_overrides", {})
    if not overrides:
        return base
    return Org(
        company=base.company,
        function=base.function,
        roles=[
            replace(r, onet_soc=overrides.get(r.role_id, r.onet_soc)) for r in base.roles
        ],
    )


# --------------------------------------------------------------------------- #
# Sidebar - org source and status
# --------------------------------------------------------------------------- #

st.sidebar.title("🪜 SkillPath")
st.sidebar.caption("Skills-based career progression, computed from O*NET.")

if not Path(DB_PATH).exists():
    st.error(
        f"O*NET database not found at `{DB_PATH}`.\n\n"
        "Build it first:  `python build.py`"
    )
    st.stop()

source = st.sidebar.radio(
    "Org structure",
    ["Use the sample Finance org", "Upload an org JSON"],
    help="The tool needs one JSON file describing the roles in a function.",
)

upload = None
if source == "Upload an org JSON":
    upload = st.sidebar.file_uploader("Org JSON", type=["json"])

demo = st.sidebar.toggle(
    "Demo view",
    value=False,
    help=(
        "One screen: pick a role, see the ladder and the callout. Everything else "
        "hidden. This is the spec's §10 money shot."
    ),
)

st.sidebar.divider()
claude = get_claude()
if claude:
    st.sidebar.success("Claude connected — matching available")
else:
    st.sidebar.warning(
        "No `ANTHROPIC_API_KEY` found. The whole app works without it; only the "
        "title→O\\*NET matching and the written summary are unavailable.\n\n"
        "Add a key to `.env` to enable them."
    )

with st.sidebar.expander("What this does and doesn't do"):
    st.markdown(
        """
**Does:** represents every role as a 68-dimension O\\*NET skill/knowledge vector,
finds the smallest skill jump to a more senior role, and derives the skills to
develop from the element-wise gap. The numbers are computed, never generated.

**Doesn't:** model headcount, budget or geography as constraints — the skill path
is aspirational, the org path is real, and the gap between them is the insight.
One function at a time; cross-function moves are out of scope.

**Data caveats:** O\\*NET is US-centric. 122 of 1016 occupations carry no ratings
and are excluded from matching entirely. Broad SOC codes leak irrelevant
descriptors — a Finance Director mapped to *Chief Executives* picks up ratings
averaged across every industry.
        """
    )

# --------------------------------------------------------------------------- #
# Load the org
# --------------------------------------------------------------------------- #

conn = get_conn()

try:
    if source == "Upload an org JSON":
        if upload is None:
            st.title("SkillPath")
            st.info("Upload an org JSON in the sidebar, or switch to the sample Finance org.")
            with st.expander("Expected format"):
                st.code(SAMPLE_ORG_PATH.read_text(encoding="utf-8"), language="json")
            st.stop()
        base_org = load_org_from_upload(upload)
    else:
        base_org = load_org(SAMPLE_ORG_PATH)
except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
    st.title("SkillPath")
    st.error(f"Could not read that org file:\n\n{exc}")
    st.stop()

st.session_state.setdefault("soc_overrides", {})
st.session_state.setdefault("matches", {})
st.session_state.setdefault("answers", {})

org = current_org(base_org)
issues = validate_org(org, conn)
blocking = errors(issues)

# Options are role_ids, not Role objects: Role is a frozen dataclass compared by
# value, so confirming a new SOC would produce a different Role and silently reset
# the user's selection. role_id is stable across overrides. Defined here rather than
# beside the section-3 picker because the demo view below uses the same two.
choices = [r.role_id for r in sorted(org.roles, key=lambda r: (r.level, r.role_id))]


def _role_label(role_id: str) -> str:
    r = org.get(role_id)
    return f"L{r.level} · {r.title}  ({r.headcount or '?'} people)"


# --------------------------------------------------------------------------- #
# Demo view - the spec's section 10 money shot
# --------------------------------------------------------------------------- #


def _demo_default(drift) -> str:
    """
    The role this demo should open on: most people behind an actual finding.

    Deliberately a rule rather than a hardcoded 'fin-analyst' - an uploaded org
    should also open on its own strongest case. Ties break on the lower level, then
    role_id, so the choice is stable. Falls back to the first role when nothing is
    flagged at all (a healthy org has no money shot, and that is a real answer).
    """
    flagged = [d for d in drift.roles if d.assessable and d.flagged]
    if not flagged:
        return choices[0]
    best = min(flagged, key=lambda d: (-d.headcount, d.role.level, d.role.role_id))
    return best.role.role_id


if demo:
    drift = team_drift(org, conn)

    # The whole point of this view is that it fits one viewport with no scrolling, so
    # it reclaims Streamlit's default top padding and skips the big st.title.
    st.markdown(
        "<style>.block-container{padding-top:2.2rem;padding-bottom:0rem;}</style>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"##### {org.company} · {org.function} · {len(org.roles)} roles · "
        f"{drift.people_total} people"
    )

    # A key of its own, NOT section 3's 'selected_role'. Sharing one key sounds tidier
    # but silently breaks the default: the toggle starts off, so the full view always
    # renders first and stamps 'selected_role' with the level-1 role before the demo
    # is ever opened - _demo_default would never fire. setdefault then seeds this key
    # once and never overrides a choice the user makes inside the demo.
    st.session_state.setdefault("demo_role", _demo_default(drift))

    demo_role_id = st.selectbox(
        "Role",
        choices,
        format_func=_role_label,
        key="demo_role",
        label_visibility="collapsed",
    )
    demo_role = org.get(demo_role_id)

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        if demo_role.onet_soc and is_rated(demo_role.onet_soc, conn):
            st.graphviz_chart(
                ladder_dot(build_ladder(org, demo_role_id, conn), conn, size="7,5.6"),
                width="stretch",
            )
        else:
            st.info(
                f"**{demo_role.title}** has no rated O\\*NET occupation, so it has no "
                "vector and no ladder. The function-level finding still holds."
            )

    with right:
        if demo_role.onet_soc and is_rated(demo_role.onet_soc, conn):
            div = divergence(org, demo_role_id, conn)
            if div.off_ladder:
                best = div.off_ladder[0]
                offered = (
                    f"<b>{div.baseline_role.title}</b> at "
                    f"{div.baseline_similarity:.2f}"
                    if div.baseline_role
                    else "nothing"
                )
                callout(
                    "gap", "◆", "These skills point somewhere you have no ladder",
                    f"<b>{demo_role.title}</b>'s skill vector sits closest to "
                    f"<b>{best.title}</b> ({best.soc_code}) at cosine "
                    f"<b>{best.similarity:.2f}</b>.<br><br>"
                    f"The best step up you offer is {offered} — "
                    f"{best.margin:+.2f} further away.",
                )
            elif not div.path.agrees:
                callout(
                    "gap", "◆", "The org chart and the skills disagree",
                    f"The chart sends <b>{demo_role.title}</b> to "
                    f"<b>{div.path.org_next.title}</b> "
                    f"({div.path.org_next_similarity:.2f}), but the smallest skill "
                    f"jump is <b>{div.path.skill_next.title}</b> "
                    f"({div.path.skill_next_similarity:.2f}).",
                )
            else:
                callout(
                    "ok", "✓", "No divergence",
                    f"<b>{demo_role.title}</b>'s best step up is also its smallest "
                    "skill jump, and nothing outside the org is closer.",
                )

        if drift.missing:
            top = drift.missing[0]
            callout(
                "gap", "▲", "And it is not one role",
                f"<b>{top.headcount} of {drift.people_total} people</b> in "
                f"{org.function} sit in roles whose skills point at "
                f"<b>{top.title}</b> ({top.soc_code}) — flagged by "
                f"{top.role_count} roles.<br><br>"
                f"<b>{drift.people_exposed} of {drift.people_total}</b> are in a role "
                "whose skills point somewhere you cannot promote them.",
            )
        elif drift.flagged:
            callout(
                "gap", "▲", "And it is not one role",
                f"<b>{drift.people_exposed} of {drift.people_total} people</b> in "
                f"{org.function} sit in a role whose published next rung is not "
                "their smallest skill jump.",
            )

    st.caption(
        "Ladder rungs and every cosine are computed from 68-dimension O\\*NET skill "
        "and knowledge vectors — no LLM touches a number. Turn off **Demo view** in "
        "the sidebar for the full analysis, the recommendations and the export."
    )
    st.stop()

st.title(org.company)
st.caption(f"{org.function} function · {len(org.roles)} roles")

# --------------------------------------------------------------------------- #
# 1. The org
# --------------------------------------------------------------------------- #

st.header("1 · The org")

table = [
    {
        "role_id": r.role_id,
        "title": r.title,
        "level": r.level,
        "reports to": r.reports_to or "— (root)",
        "headcount": r.headcount,
        "O*NET": r.onet_soc or "—",
        "occupation": occupation_title(r.onet_soc, conn) if r.onet_soc else "—",
        "usable": "yes" if r.onet_soc and is_rated(r.onet_soc, conn) else "NO",
    }
    for r in sorted(org.roles, key=lambda r: (r.level, r.role_id))
]
st.dataframe(table, hide_index=True, width="stretch")

if issues:
    with st.expander(f"Validation: {len(issues)} issue(s), {len(blocking)} blocking", expanded=bool(blocking)):
        for i in issues:
            (st.error if i.severity == "error" else st.warning)(str(i))
else:
    callout(
        "ok", "✓", "Org validates cleanly",
        "Every role maps to a rated O*NET occupation, so every rung has a vector.",
    )

# --------------------------------------------------------------------------- #
# 2. Where this function's skills are drifting  (Phase 6)
# --------------------------------------------------------------------------- #

st.header("2 · Where this function's skills are drifting")

drift = team_drift(org, conn)

st.caption(
    f"{len(drift.roles)} roles · {drift.people_total} people · "
    f"{len(drift.assessed)} assessed, {len(drift.unassessable)} not assessable"
)

if not drift.flagged:
    callout(
        "ok", "✓", "No drift",
        f"Every assessed role in {org.function} already has its smallest available "
        "skill jump on the published ladder, and no O*NET occupation outside this "
        "org is closer for anyone.",
    )
else:
    if drift.missing:
        top = drift.missing[0]
        callout(
            "gap", "▲",
            f"{top.headcount} of {drift.people_total} people point at a ladder "
            "you do not have",
            f"Their skills sit closest to <b>{top.title}</b> ({top.soc_code}) — an "
            f"occupation this function has <b>no ladder for</b>. It is an off-ladder "
            f"match for {top.role_count} role(s): "
            + ", ".join(f"<i>{r.title}</i>" for r in top.roles)
            + f".<br><br>Across the function, <b>{drift.people_exposed} of "
            f"{drift.people_total}</b> people are in roles whose skills point "
            "somewhere you cannot promote them.",
        )
        st.dataframe(
            [
                {
                    "occupation": m.title,
                    "O*NET": m.soc_code,
                    "people": m.headcount,
                    "roles": m.role_count,
                    "pointing at it": ", ".join(r.title for r in m.roles),
                    "best cosine": m.best_similarity,
                    "best margin": m.best_margin,
                }
                for m in drift.missing
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Ranked by the people behind each destination, not by cosine — a track "
            "three roles point at outranks a closer one only a single role points at. "
            "Each role's margin is measured against **its own** best step up, so "
            "margins compare within a destination but are never summed across the "
            "function; headcount is the unit that aggregates cleanly."
        )

    if drift.path_disagreements:
        with st.expander(
            f"{len(drift.path_disagreements)} role(s) where the org chart and the "
            "skill ladder disagree"
        ):
            for d in drift.path_disagreements:
                path = d.divergence.path
                st.markdown(
                    f"- **{d.role.title}** ({d.headcount} people): the chart says "
                    f"*{path.org_next.title}* (`{path.org_next_similarity:.2f}`), the "
                    f"skills say *{path.skill_next.title}* "
                    f"(`{path.skill_next_similarity:.2f}`), `{path.margin:+.2f}` closer."
                )

with st.expander("Per-role detail, including roles that could not be assessed"):
    st.dataframe(
        [
            {
                "role": d.role.title,
                "level": d.role.level,
                "people": d.headcount,
                "assessed": "yes" if d.assessable else "no",
                "points off-ladder": (
                    d.divergence.off_ladder[0].title
                    if d.assessable and d.divergence.off_ladder
                    else ""
                ),
                "margin": round(d.top_margin, 4) if d.assessable else None,
                "why not": d.reason or "",
            }
            for d in drift.roles
        ],
        hide_index=True,
        width="stretch",
    )
    if drift.headcount_assumed:
        st.caption(
            "Some roles carry no `headcount` in the org file and were counted as "
            "1 person each — never 0, which would erase them from the totals."
        )

# --------------------------------------------------------------------------- #
# 3. Pick a role
# --------------------------------------------------------------------------- #

st.header("3 · Pick a role")

role_id = st.selectbox(
    "Role to map a progression for",
    choices,
    format_func=_role_label,
    key="selected_role",
)
role = org.get(role_id)

chain = org_path(org, role.role_id)
st.markdown(
    "**Org path** (what the org chart says): "
    + "  →  ".join(f"`{r.title}`" for r in chain)
)

# --------------------------------------------------------------------------- #
# 4. Confirm the O*NET match
# --------------------------------------------------------------------------- #

st.header("4 · Confirm the O*NET match")

usable = bool(role.onet_soc) and is_rated(role.onet_soc, conn)
if usable:
    st.markdown(
        f"**{role.title}** is currently mapped to "
        f"`{role.onet_soc}` — *{occupation_title(role.onet_soc, conn)}*."
    )
else:
    st.warning(f"**{role.title}** has no usable O*NET code yet. Match it below to continue.")

if not claude:
    st.info("Matching needs an `ANTHROPIC_API_KEY`. The existing mapping above is still used.")
else:
    from src.matching import match_role

    col_a, col_b = st.columns([1, 3])
    with col_a:
        run = st.button("Match with Claude", type="primary" if not usable else "secondary")
    with col_b:
        st.caption(
            "Claude picks from a shortlist generated by a local keyword search over "
            "rated occupations only — it cannot invent a code."
        )

    if run:
        with st.spinner("Searching O*NET and asking Claude…"):
            try:
                prior = st.session_state["answers"].get(role.role_id) or None
                st.session_state["matches"][role.role_id] = match_role(
                    role.title, role.description, conn, claude, answers=prior
                )
            except Exception as exc:  # noqa: BLE001 - surface any API/validation failure
                st.error(f"Matching failed: {exc}")

    match = st.session_state["matches"].get(role.role_id)
    if match:
        if match.warning:
            st.warning(match.warning)

        badge = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(match.proposal.confidence, "⚪")
        st.markdown(
            f"{badge} Claude proposes **`{match.soc_code}` — {match.matched_title}** "
            f"({match.proposal.confidence} confidence)"
        )
        st.caption(match.proposal.reasoning)

        options = [h.soc_code for h in match.candidates]
        chosen = st.selectbox(
            "Confirm, or pick a different occupation from the shortlist",
            options,
            index=options.index(match.soc_code) if match.soc_code in options else 0,
            format_func=lambda c: f"{c} — {match.title_of.get(c, c)}",
            key=f"confirm_{role.role_id}",
        )
        if st.button("Use this occupation", key=f"apply_{role.role_id}"):
            st.session_state["soc_overrides"][role.role_id] = chosen
            st.rerun()

        st.subheader("Clarifying questions")
        st.caption(
            "These exist to disambiguate the occupational match. Answer them and "
            "re-run to let Claude revise its pick."
        )
        replies: list[tuple[str, str]] = []
        for n, q in enumerate(match.proposal.questions):
            answer = st.radio(
                q.question,
                q.options,
                key=f"q_{role.role_id}_{n}",
                help=q.why,
                horizontal=False,
            )
            replies.append((q.question, answer))

        if st.button("Re-match using my answers", key=f"refine_{role.role_id}"):
            st.session_state["answers"][role.role_id] = replies
            with st.spinner("Re-matching with your answers…"):
                try:
                    st.session_state["matches"][role.role_id] = match_role(
                        role.title, role.description, conn, claude, answers=replies
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Re-match failed: {exc}")
            st.rerun()

# --------------------------------------------------------------------------- #
# 5. The progression map
# --------------------------------------------------------------------------- #

st.header("5 · Skill-natural progression")

if not usable:
    st.info("Confirm an O*NET match for this role to compute its progression.")
    st.stop()

ladder = build_ladder(org, role.role_id, conn)

if not ladder.steps:
    st.info(
        f"**{role.title}** is at the top of this org's levels — there is no more "
        "senior role to progress into."
    )
else:
    left, right = st.columns([1, 1])

    with left:
        st.graphviz_chart(ladder_dot(ladder, conn), width="stretch")

    with right:
        for step in ladder.steps:
            with st.container(border=True):
                st.markdown(
                    f"**{step.from_role.title} → {step.to_role.title}**  \n"
                    f"cosine `{step.similarity:.3f}`"
                    + ("  ·  same occupation" if step.same_occupation else "")
                )
                if step.note:
                    st.caption(step.note)

                recs = step.recommendations
                if not recs["skills"] and not recs["knowledge"]:
                    st.caption("No development recommendations available for this step.")
                    continue
                if recs["skills"]:
                    st.markdown("*Skills to develop*")
                    for g in recs["skills"]:
                        st.markdown(
                            f"- {g['element_name']} — level "
                            f"`{g['lv_current']:.1f} → {g['lv_next']:.1f}` (+{g['delta']:.2f})"
                        )
                if recs["knowledge"]:
                    st.markdown("*Knowledge to build*")
                    for g in recs["knowledge"]:
                        st.markdown(
                            f"- {g['element_name']} — level "
                            f"`{g['lv_current']:.1f} → {g['lv_next']:.1f}` (+{g['delta']:.2f})"
                        )

    with st.expander("All senior roles ranked by skill adjacency"):
        st.dataframe(
            [
                {
                    "role": c.role.title,
                    "level": c.role.level,
                    "O*NET": c.role.onet_soc,
                    "cosine": round(c.similarity, 4),
                    "same occupation": "yes" if c.same_occupation else "",
                }
                for c in rank_candidates(org, role.role_id, conn)
            ],
            hide_index=True,
            width="stretch",
        )

# --------------------------------------------------------------------------- #
# 6. The divergence flag  (Phase 5 - the differentiator)
# --------------------------------------------------------------------------- #

st.header("6 · Org-vs-skill divergence")

div = divergence(org, role.role_id, conn)

if not div.flagged:
    callout(
        "ok", "✓", "No divergence",
        f"The best step up {org.company} offers {role.title} is also the smallest "
        "skill jump available, and no O*NET occupation outside this org is closer. "
        "The published ladder matches where these skills point.",
    )
else:
    if div.off_ladder:
        best = div.off_ladder[0]
        offered = (
            f"<b>{div.baseline_role.title}</b> at cosine "
            f"{div.baseline_similarity:.2f}"
            if div.baseline_role
            else "nothing"
        )
        callout(
            "gap", "◆", "These skills point somewhere you have no ladder",
            f"<b>{role.title}</b>'s skill vector sits closest to "
            f"<b>{best.title}</b> ({best.soc_code}) at cosine "
            f"<b>{best.similarity:.2f}</b> — a track this org has "
            f"<b>no ladder for</b>. The best step up you do offer is {offered}, "
            f"{best.margin:+.2f} further away.<br><br>"
            "These people's skills point somewhere you cannot promote them — "
            "retention risk, or an opportunity to build the lateral track.",
        )
        st.caption(
            "Ranked against every rated O\\*NET occupation, excluding those this org "
            "already has a role for. “Margin” is how much closer each one is than the "
            "org's own best step up — the comparison is relative because absolute "
            "cosine over O\\*NET vectors is compressed into a narrow high band."
        )
        st.dataframe(
            [
                {
                    "occupation": m.title,
                    "O*NET": m.soc_code,
                    "cosine": m.similarity,
                    "margin vs offered": m.margin,
                }
                for m in div.off_ladder
            ],
            hide_index=True,
            width="stretch",
        )

    if not div.path.agrees:
        callout(
            "gap", "◆", "The org chart and the skill ladder disagree",
            f"{org.company} sends {role.title} to "
            f"<b>{div.path.org_next.title}</b> "
            f"(cosine {div.path.org_next_similarity:.2f}), but the smallest skill "
            f"jump is <b>{div.path.skill_next.title}</b> "
            f"(cosine {div.path.skill_next_similarity:.2f}), "
            f"{div.path.margin:+.2f} closer. The promotion you offer is a bigger "
            "jump than a move you don't.",
        )


# --------------------------------------------------------------------------- #
# 7. Export
# --------------------------------------------------------------------------- #

st.header("7 · Export")


def build_markdown(narration: str | None = None) -> str:
    lines = [
        f"# Career progression — {role.title}",
        f"*{org.company} · {org.function} · generated {date.today().isoformat()}*",
        "",
        f"**Current role:** {role.title} (level {role.level}, "
        f"{role.headcount or '?'} people)  ",
        f"**O\\*NET occupation:** `{role.onet_soc}` — {occupation_title(role.onet_soc, conn)}",
        "",
        "**Org path (what the org chart says):** "
        + " → ".join(r.title for r in chain),
        "",
        "## Skill-natural progression",
        "",
    ]
    if not ladder.steps:
        lines.append("_Top of the ladder — no more senior role in this org._")
    for step in ladder.steps:
        lines.append(
            f"### {step.from_role.title} → {step.to_role.title}  (cosine {step.similarity:.3f})"
        )
        if step.note:
            lines.append(f"> {step.note}")
        recs = step.recommendations
        if recs["skills"]:
            lines.append("")
            lines.append("**Skills to develop**")
            lines += [
                f"- {g['element_name']} ({g['lv_current']:.1f} → {g['lv_next']:.1f})"
                for g in recs["skills"]
            ]
        if recs["knowledge"]:
            lines.append("")
            lines.append("**Knowledge to build**")
            lines += [
                f"- {g['element_name']} ({g['lv_current']:.1f} → {g['lv_next']:.1f})"
                for g in recs["knowledge"]
            ]
        lines.append("")

    lines.append("## Function-level drift")
    lines.append("")
    if not drift.flagged:
        lines.append(
            f"_No drift across {org.function}. Every assessed role's best step up is "
            "already its smallest available skill jump._"
        )
        lines.append("")
    else:
        lines.append(
            f"{drift.people_exposed} of {drift.people_total} people in {org.function} "
            "sit in roles whose skills point at an occupation this function has no "
            "ladder for."
        )
        lines.append("")
        if drift.missing:
            lines.append("| Missing ladder | O\\*NET | People | Roles | Best cosine |")
            lines.append("|---|---|---|---|---|")
            lines += [
                f"| {m.title} | `{m.soc_code}` | {m.headcount} | {m.role_count} | "
                f"{m.best_similarity:.4f} |"
                for m in drift.missing
            ]
            lines.append("")
        lines.append(
            "_Ranked by the people behind each destination. Each role's margin is "
            "measured against its own best step up, so margins are not summed across "
            "the function._"
        )
        lines.append("")

    lines.append("## Org-vs-skill divergence")
    lines.append("")
    if not div.flagged:
        lines.append(
            "_No divergence. The best step up this org offers is also the smallest "
            "skill jump, and no outside occupation is closer._"
        )
        lines.append("")
    else:
        if div.off_ladder:
            best = div.off_ladder[0]
            offered = (
                f"{div.baseline_role.title} (cosine {div.baseline_similarity:.2f})"
                if div.baseline_role
                else "nothing"
            )
            lines.append(
                f"**{role.title}**'s skills sit closest to **{best.title}** "
                f"(`{best.soc_code}`, cosine {best.similarity:.2f}) — a track this org "
                f"has no ladder for. The best step up offered is {offered}, "
                f"{best.margin:+.2f} further away."
            )
            lines.append("")
            lines.append("| Occupation | O\\*NET | Cosine | Margin vs offered |")
            lines.append("|---|---|---|---|")
            lines += [
                f"| {m.title} | `{m.soc_code}` | {m.similarity:.4f} | {m.margin:+.4f} |"
                for m in div.off_ladder
            ]
            lines.append("")
        if not div.path.agrees:
            lines.append(
                f"**Path divergence.** The org chart sends {role.title} to "
                f"{div.path.org_next.title} (cosine {div.path.org_next_similarity:.2f}), "
                f"but the smallest skill jump is {div.path.skill_next.title} "
                f"(cosine {div.path.skill_next_similarity:.2f}), "
                f"{div.path.margin:+.2f} closer."
            )
            lines.append("")

    answers = st.session_state["answers"].get(role.role_id)
    if answers:
        lines.append("## Context supplied")
        lines += [f"- **{q}** {a}" for q, a in answers]
        lines.append("")

    if narration:
        lines.append("## Summary")
        lines.append(narration)
        lines.append("")

    lines += [
        "---",
        "",
        "Similarities are cosine distances between 68-dimension O\\*NET skill and "
        "knowledge vectors (value = Level × Importance). Development areas are the "
        "element-wise Level deltas to the next role. Computed, not generated.",
        "",
        "Includes information from the O\\*NET 30.3 Database by USDOL/ETA, used under "
        "CC BY 4.0. O\\*NET® is a trademark of USDOL/ETA.",
    ]
    return "\n".join(lines)


narration = st.session_state.get(f"narration_{role.role_id}")

if claude and ladder.steps:
    if st.button("Write a plain-language summary with Claude"):
        from src.engine import format_ladder
        from src.matching import narrate_summary

        with st.spinner("Writing…"):
            try:
                st.session_state[f"narration_{role.role_id}"] = narrate_summary(
                    org.company,
                    org.function,
                    format_ladder(ladder, conn),
                    st.session_state["answers"].get(role.role_id),
                    claude,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not write the summary: {exc}")
        st.rerun()

if narration:
    st.markdown(narration)

markdown = build_markdown(narration)
st.download_button(
    "Download one-page summary (Markdown)",
    data=markdown,
    file_name=f"skillpath-{role.role_id}-{date.today().isoformat()}.md",
    mime="text/markdown",
)

with st.expander("Preview the export"):
    st.markdown(markdown)
