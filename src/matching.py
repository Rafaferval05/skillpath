"""
SkillPath Phase 2 - title -> O*NET SOC matching (the ONLY module that calls an LLM).

The division of labour matters and is deliberate:

  * Candidate generation is DETERMINISTIC - a local keyword search over the
    occupation table, restricted to occupations that actually have ratings.
  * Claude PICKS from that shortlist and explains itself. It cannot invent a SOC
    code: anything outside the shortlist is rejected by `_validate_pick`.
  * Claude never computes an adjacency number. That stays in vectors.py/engine.py.

Public API
----------
    search_occupations(query, conn, limit) -> list[OccupationHit]
    match_role(title, description, conn, client, answers) -> RoleMatch
    narrate_summary(org, ladder, answers, client) -> str
    get_client() -> anthropic.Anthropic | None

WHY LOCAL SEARCH INSTEAD OF THE O*NET WEB API
---------------------------------------------
The spec calls for O*NET's keyword-search endpoint. We already have the whole
occupation table in SQLite, so searching locally is offline, rate-limit free, and
- the real reason - it can filter to RATED occupations up front. O*NET's own API
happily returns codes like 13-2051.00 that have no skill ratings and therefore no
vector, which would push the failure downstream into the math layer.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

from src.vectors import DB_PATH

MODEL = "claude-opus-5"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Seniority and filler words carry org-level meaning, not occupational meaning.
# "Senior Financial Analyst" and "Financial Analyst" are the same O*NET occupation.
_STOPWORDS = {
    "senior", "sr", "junior", "jr", "lead", "principal", "staff", "chief", "head",
    "associate", "assistant", "deputy", "global", "regional", "corporate", "group",
    "i", "ii", "iii", "iv", "v", "1", "2", "3", "of", "and", "the", "for", "a", "an",
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# --------------------------------------------------------------------------- #
# Deterministic candidate generation
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OccupationHit:
    soc_code: str
    title: str
    description: str
    score: float

    def short_description(self, n: int = 240) -> str:
        d = (self.description or "").strip()
        return d if len(d) <= n else d[: n - 1].rsplit(" ", 1)[0] + "..."


def _tokens(text: str) -> list[str]:
    """Lowercase word tokens with seniority/filler words removed."""
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS]


@lru_cache(maxsize=4)
def _search_index(rated_only: bool = True) -> tuple:
    """
    Build (and cache) the TF-IDF index over occupation titles and descriptions.

    Character n-grams rather than whole words, because job titles and O*NET titles
    differ morphologically far more than they differ semantically: "Finance" vs
    "Financial", "Analyst" vs "Analysts", "Accounts" vs "Accounting". Word-level
    matching misses all three; 3-5 char n-grams catch them without a stemmer.

    Built once per process - 894 rows is small, but this runs on every keystroke
    in the UI.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    conn = sqlite3.connect(DB_PATH)
    try:
        sql = "SELECT o.soc_code, o.title, o.description FROM occupation o"
        if rated_only:
            sql += " WHERE EXISTS (SELECT 1 FROM ratings r WHERE r.soc_code = o.soc_code)"
        rows = conn.execute(sql + " ORDER BY o.soc_code").fetchall()
    finally:
        conn.close()

    titles = [r[1] or "" for r in rows]
    descriptions = [r[2] or "" for r in rows]

    title_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    title_matrix = title_vec.fit_transform(titles)

    desc_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    desc_matrix = desc_vec.fit_transform(descriptions)

    return rows, title_vec, title_matrix, desc_vec, desc_matrix


def search_occupations(
    query: str,
    conn: sqlite3.Connection | None = None,
    limit: int = 12,
    rated_only: bool = True,
) -> list[OccupationHit]:
    """
    Keyword-search the local occupation table. Pure Python, no LLM, no network.

    Scores each occupation as a blend of fuzzy title similarity (dominant),
    fuzzy description similarity, and an exact word-overlap bonus so a literal
    title hit always outranks a merely n-gram-similar one.

    `rated_only` keeps unrated codes out of the shortlist entirely, so Claude can
    only pick something the math layer can actually build a vector for. `conn` is
    accepted for API symmetry with the rest of the codebase but is unused - the
    index is cached process-wide.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    q_tokens = _tokens(query)
    if not q_tokens:
        return []
    q_norm = " ".join(q_tokens)

    rows, title_vec, title_matrix, desc_vec, desc_matrix = _search_index(rated_only)

    title_sim = cosine_similarity(title_vec.transform([q_norm]), title_matrix)[0]
    desc_sim = cosine_similarity(desc_vec.transform([q_norm]), desc_matrix)[0]

    q_set = set(q_tokens)
    hits: list[OccupationHit] = []
    for i, (soc, title, description) in enumerate(rows):
        score = 75.0 * float(title_sim[i]) + 20.0 * float(desc_sim[i])

        t_set = set(_tokens(title))
        if t_set:
            overlap = q_set & t_set
            if overlap:
                # A literal word hit is stronger evidence than n-gram overlap.
                score += 25.0 * (len(overlap) / len(q_set))
            if t_set == q_set:
                score += 40.0

        if score > 1.0:
            hits.append(OccupationHit(soc, title, description or "", round(score, 2)))

    hits.sort(key=lambda h: (-h.score, h.soc_code))
    return hits[:limit]


# --------------------------------------------------------------------------- #
# The LLM layer
# --------------------------------------------------------------------------- #

class ClarifyingQuestion(BaseModel):
    """A question that would change or confirm the occupational match."""

    question: str = Field(description="The question to put to the HR user.")
    why: str = Field(description="One short sentence on what the answer would change.")
    options: list[str] = Field(
        description="2 to 4 short, mutually exclusive answer options.",
        min_length=2,
        max_length=4,
    )


class MatchProposal(BaseModel):
    """Claude's structured pick from the shortlist."""

    soc_code: str = Field(description="The chosen O*NET-SOC code, copied EXACTLY from the shortlist.")
    confidence: str = Field(description="One of: high, medium, low.")
    reasoning: str = Field(description="Two sentences at most, in plain HR language.")
    runner_up_soc: str | None = Field(
        default=None,
        description="The next-best code from the shortlist, or null if nothing else fits.",
    )
    questions: list[ClarifyingQuestion] = Field(
        description="Exactly 2 or 3 clarifying questions.", min_length=2, max_length=3
    )


@dataclass
class RoleMatch:
    """A proposal plus the shortlist it was drawn from, ready for a human to confirm."""

    proposal: MatchProposal
    candidates: list[OccupationHit]
    title_of: dict[str, str]
    warning: str | None = None

    @property
    def soc_code(self) -> str:
        return self.proposal.soc_code

    @property
    def matched_title(self) -> str:
        return self.title_of.get(self.soc_code, self.soc_code)


SYSTEM_PROMPT = """\
You map internal HR job titles onto O*NET occupations for a career-pathing tool.

Rules you must follow:
- Choose ONLY from the numbered shortlist you are given. Never output a SOC code \
that is not on it, even if you believe a better one exists elsewhere in O*NET.
- O*NET occupations describe the OCCUPATION, not the seniority. "Senior Financial \
Analyst" and "Financial Analyst" usually map to the same occupation; do not reach \
for a management occupation just because a title says Senior or Lead. Map to a \
management occupation only when the role genuinely manages people or a function.
- Your clarifying questions must be ones whose answers would actually change the \
match or confirm a genuinely uncertain one. Ask about the substance of the work: \
what the person does day to day, whether they manage people, which domain they \
sit in. Do not ask about things you were already told.
- Write for an HR business partner, not a data scientist. No jargon, no hedging.\
"""


def get_client():
    """
    Build an Anthropic client, or return None if no key is configured.

    Reads .env (git-ignored) if python-dotenv is installed, so the key never has
    to be hardcoded or exported by hand.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None

    import anthropic

    return anthropic.Anthropic()


def _format_shortlist(candidates: list[OccupationHit]) -> str:
    lines = []
    for i, h in enumerate(candidates, start=1):
        lines.append(f"{i}. {h.soc_code}  {h.title}\n   {h.short_description()}")
    return "\n".join(lines)


def _validate_pick(proposal: MatchProposal, candidates: list[OccupationHit]) -> str | None:
    """
    Enforce that the pick came from the shortlist. Returns a warning, or None.

    This is the guard that keeps a hallucinated SOC code out of the math layer.
    """
    allowed = {h.soc_code for h in candidates}

    # Sanitize the runner-up BEFORE the primary check. A bad primary pick returns
    # early below, and a hallucinated runner-up must not ride along on that path -
    # it is offered to the user as a one-click alternative in the UI.
    if proposal.runner_up_soc and proposal.runner_up_soc not in allowed:
        proposal.runner_up_soc = None

    if proposal.soc_code not in allowed:
        fallback = candidates[0].soc_code
        bad = proposal.soc_code
        proposal.soc_code = fallback
        proposal.confidence = "low"
        return (
            f"Claude returned {bad}, which was not on the shortlist. Fell back to the "
            f"top keyword hit ({fallback}) - confirm this manually."
        )
    if proposal.confidence not in {"high", "medium", "low"}:
        proposal.confidence = "medium"
    return None


def match_role(
    title: str,
    description: str | None,
    conn: sqlite3.Connection,
    client,
    answers: list[tuple[str, str]] | None = None,
    limit: int = 12,
) -> RoleMatch:
    """
    Propose an O*NET occupation for a job title, with 2-3 clarifying questions.

    `answers` re-runs the match with the user's replies folded in - that is what
    the clarifying questions are FOR, so a second pass can revise the pick.
    Raises ValueError if the local search finds nothing to choose from.
    """
    candidates = search_occupations(title, conn, limit=limit)
    if not candidates and description:
        candidates = search_occupations(description, conn, limit=limit)
    if not candidates:
        raise ValueError(
            f"No O*NET occupations matched '{title}'. Try a more conventional job title."
        )

    parts = [f"Internal job title: {title}"]
    if description:
        parts.append(f"What the role does: {description}")
    if answers:
        replies = "\n".join(f"- {q}\n  Answer: {a}" for q, a in answers)
        parts.append(
            "The HR user answered your earlier clarifying questions as follows. "
            "Revise your pick if these change the picture, and ask NEW questions "
            "if anything is still genuinely ambiguous:\n" + replies
        )
    parts.append("Shortlist of candidate O*NET occupations:\n" + _format_shortlist(candidates))
    parts.append("Pick the single best occupation for this role.")

    response = client.messages.parse(
        model=MODEL,
        max_tokens=8000,
        system=SYSTEM_PROMPT,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
        output_format=MatchProposal,
    )
    proposal = response.parsed_output
    warning = _validate_pick(proposal, candidates)

    return RoleMatch(
        proposal=proposal,
        candidates=candidates,
        title_of={h.soc_code: h.title for h in candidates},
        warning=warning,
    )


def narrate_summary(
    company: str,
    function: str,
    ladder_text: str,
    answers: list[tuple[str, str]] | None,
    client,
) -> str:
    """
    Turn the computed ladder into a short plain-language brief for an HR reader.

    Claude is explicitly told the numbers are given, not negotiable - it narrates
    what the math produced and must not invent a different progression.
    """
    context = [
        f"Company: {company}",
        f"Function: {function}",
        "",
        "The career ladder below was computed from O*NET skill vectors. The roles, "
        "the cosine similarities and the recommended descriptors are FIXED - narrate "
        "them, do not change them, do not add roles or skills that are not listed.",
        "",
        ladder_text,
    ]
    if answers:
        context.append("")
        context.append(
            "Context the HR user supplied:\n"
            + "\n".join(f"- {q} -> {a}" for q, a in answers)
        )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=(
            "You write short, concrete career-development briefs for HR business "
            "partners. Plain language, no jargon, no bullet-point padding. Around "
            "200 words. Structure: what this progression looks like, what the person "
            "needs to build at each step, and one honest caveat about the data."
        ),
        messages=[{"role": "user", "content": "\n".join(context)}],
    )
    return "\n".join(b.text for b in response.content if b.type == "text").strip()
