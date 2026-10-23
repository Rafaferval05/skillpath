"""
SkillPath Phase 0 - core skill-adjacency math (pure, deterministic, NO LLM).

Public API
----------
    build_vector(soc_code)                  -> np.ndarray   (68-dim, value = LV * IM)
    similarity(soc_a, soc_b)                -> float        (cosine, 0..1)
    skill_gap(soc_current, soc_next)        -> list[dict]   (top-5 trainable LV deltas)
    skill_gap_split(soc_current, soc_next)  -> dict         (top-3 skills + top-3 knowledge)
    is_rated(soc_code)                      -> bool         (does this SOC have a vector?)
    occupation_title(soc_code)              -> str

All three functions read from the SQLite DB built by onet_loader.py. They accept
an optional open sqlite3 connection so a caller (e.g. the validation script) can
reuse one connection across many calls.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "onet.db"


# --------------------------------------------------------------------------- #
# Connection / dimension helpers
# --------------------------------------------------------------------------- #

def _connect(conn: sqlite3.Connection | None = None) -> tuple[sqlite3.Connection, bool]:
    """Return (connection, owned). 'owned' is True when we opened it and must close it."""
    if conn is not None:
        return conn, False
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"O*NET database not found at {DB_PATH}. "
            "Build it first:  python -m src.onet_loader   (or run build.py)."
        )
    return sqlite3.connect(DB_PATH), True


@lru_cache(maxsize=1)
def _descriptor_index() -> tuple[tuple[str, str, str], ...]:
    """
    Canonical, fixed ordering of the descriptor axes.

    Returns a tuple of (element_id, element_name, descriptor_type), sorted by
    element_id so the vector dimensions are stable and comparable across roles.
    Cached because it never changes for a given DB.
    """
    conn, owned = _connect()
    try:
        rows = conn.execute(
            "SELECT element_id, element_name, descriptor_type FROM descriptor ORDER BY element_id"
        ).fetchall()
    finally:
        if owned:
            conn.close()
    if not rows:
        raise RuntimeError("No descriptors found in DB - was the loader run?")
    return tuple((r[0], r[1], r[2]) for r in rows)


def descriptor_ids() -> list[str]:
    """Ordered element_ids defining the vector space (length == number of dims)."""
    return [eid for eid, _name, _type in _descriptor_index()]


def rated_socs(conn: sqlite3.Connection | None = None) -> list[str]:
    """Every SOC that has ratings, sorted - the occupations a vector exists for."""
    conn, owned = _connect(conn)
    try:
        return [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT soc_code FROM ratings ORDER BY soc_code"
            ).fetchall()
        ]
    finally:
        if owned:
            conn.close()


# Cache key includes the DB's mtime + size, so a rebuilt onet.db invalidates the
# matrix instead of serving vectors from the previous build.
_MATRIX_CACHE: dict[tuple, tuple[list[str], np.ndarray]] = {}


def occupation_matrix(
    conn: sqlite3.Connection | None = None,
) -> tuple[list[str], np.ndarray]:
    """
    Every rated occupation as one L2-normalised matrix: (socs, unit_rows).

    Phase 5 scores one role against all ~894 rated occupations. Doing that with
    similarity() means ~1,800 per-call SQL queries; building the matrix once and
    taking a dot product is the same arithmetic in a fraction of the time.

    Because rows are unit vectors, `unit_rows @ unit_rows[i]` IS the vector of
    cosine similarities against occupation i - identical to similarity(), which
    normalises the same way.
    """
    key = (str(DB_PATH), DB_PATH.stat().st_mtime_ns, DB_PATH.stat().st_size)
    cached = _MATRIX_CACHE.get(key)
    if cached is not None:
        return cached

    conn, owned = _connect(conn)
    try:
        socs = rated_socs(conn)
        matrix = np.vstack([build_vector(s, conn) for s in socs])
    finally:
        if owned:
            conn.close()

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # a zero vector stays zero rather than dividing by 0
    unit = matrix / norms

    _MATRIX_CACHE.clear()  # only ever hold the current build
    _MATRIX_CACHE[key] = (socs, unit)
    return socs, unit


# --------------------------------------------------------------------------- #
# Per-role score extraction
# --------------------------------------------------------------------------- #

def _role_scores(soc_code: str, conn: sqlite3.Connection) -> dict[str, dict[str, float]]:
    """
    Return {element_id: {'IM': float, 'LV': float}} for one role.

    Only descriptors the role actually has are present; callers treat any
    missing descriptor / missing scale as 0 (handled gracefully downstream).
    """
    rows = conn.execute(
        "SELECT element_id, scale_id, data_value FROM ratings WHERE soc_code = ?",
        (soc_code,),
    ).fetchall()
    scores: dict[str, dict[str, float]] = {}
    for element_id, scale_id, value in rows:
        scores.setdefault(element_id, {})[scale_id] = value
    return scores


def _role_exists(soc_code: str, conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM ratings WHERE soc_code = ? LIMIT 1", (soc_code,)).fetchone()
        is not None
    )


def is_rated(soc_code: str, conn: sqlite3.Connection | None = None) -> bool:
    """
    True when this SOC has skill/knowledge ratings, i.e. a vector can be built.

    O*NET 30.3 rates only 894 of 1016 occupations - the spec's headline
    13-2051.00 (Financial and Investment Analysts) is among the unrated. Callers
    that accept a SOC from outside (an uploaded org, a title->SOC match) must
    check this before handing the code to build_vector / similarity, which raise.
    """
    conn, owned = _connect(conn)
    try:
        return _role_exists(soc_code, conn)
    finally:
        if owned:
            conn.close()


def occupation_title(soc_code: str, conn: sqlite3.Connection | None = None) -> str:
    """Human-readable title for a SOC code (falls back to the code itself)."""
    conn, owned = _connect(conn)
    try:
        row = conn.execute(
            "SELECT title FROM occupation WHERE soc_code = ?", (soc_code,)
        ).fetchone()
    finally:
        if owned:
            conn.close()
    return row[0] if row else soc_code


# --------------------------------------------------------------------------- #
# 1) build_vector
# --------------------------------------------------------------------------- #

def build_vector(soc_code: str, conn: sqlite3.Connection | None = None) -> np.ndarray:
    """
    Build the role's skill vector over the fixed 68-descriptor space.

    value[i] = LV[i] * IM[i]   (Level weighted by Importance)

    Missing descriptors (or a descriptor missing either scale) contribute 0.
    Raises ValueError if the role has no skill/knowledge ratings at all.
    """
    conn, owned = _connect(conn)
    try:
        if not _role_exists(soc_code, conn):
            raise ValueError(
                f"No skill/knowledge ratings found for SOC '{soc_code}'. "
                "Check the code (O*NET-SOC, e.g. '13-2051.00')."
            )
        scores = _role_scores(soc_code, conn)
        ids = descriptor_ids()
        vec = np.zeros(len(ids), dtype=float)
        for i, element_id in enumerate(ids):
            s = scores.get(element_id)
            if not s:
                continue  # descriptor absent for this role -> 0
            lv = s.get("LV", 0.0)
            im = s.get("IM", 0.0)
            vec[i] = lv * im
        return vec
    finally:
        if owned:
            conn.close()


# --------------------------------------------------------------------------- #
# 2) similarity
# --------------------------------------------------------------------------- #

def similarity(soc_a: str, soc_b: str, conn: sqlite3.Connection | None = None) -> float:
    """
    Cosine similarity between two role vectors, in [0, 1].

    Vector entries are non-negative (LV * IM >= 0), so cosine is bounded to
    [0, 1]: 0 -> unrelated skill profile, 1 -> identical.
    """
    conn, owned = _connect(conn)
    try:
        va = build_vector(soc_a, conn).reshape(1, -1)
        vb = build_vector(soc_b, conn).reshape(1, -1)
        na = float(np.linalg.norm(va))
        nb = float(np.linalg.norm(vb))
        if na == 0.0 or nb == 0.0:
            return 0.0
        sim = float(cosine_similarity(va, vb)[0, 0])
        # Guard against tiny floating-point excursions outside [0, 1].
        return max(0.0, min(1.0, sim))
    finally:
        if owned:
            conn.close()


# --------------------------------------------------------------------------- #
# 3) skill_gap  (single ranked list)  +  skill_gap_split  (skills vs knowledge)
# --------------------------------------------------------------------------- #

def _all_gaps(
    soc_current: str,
    soc_next: str,
    conn: sqlite3.Connection,
) -> list[dict]:
    """
    Every descriptor with a POSITIVE Level delta for current -> next, sorted.

    gap[i] = LV_next[i] - LV_current[i]   (raw Level delta, NOT weighted by IM)

    Sorted by delta descending, tie-broken on element_id so output is
    deterministic. Shared by skill_gap() and skill_gap_split(); neither should
    re-implement this loop.
    """
    for soc in (soc_current, soc_next):
        if not _role_exists(soc, conn):
            raise ValueError(f"No skill/knowledge ratings found for SOC '{soc}'.")

    cur = _role_scores(soc_current, conn)
    nxt = _role_scores(soc_next, conn)

    gaps = []
    for element_id, name, dtype in _descriptor_index():
        lv_cur = cur.get(element_id, {}).get("LV", 0.0)
        lv_nxt = nxt.get(element_id, {}).get("LV", 0.0)
        delta = lv_nxt - lv_cur
        if delta > 0:
            gaps.append(
                {
                    "element_id": element_id,
                    "element_name": name,
                    "descriptor_type": dtype,
                    "lv_current": round(lv_cur, 2),
                    "lv_next": round(lv_nxt, 2),
                    "delta": round(delta, 2),
                }
            )

    gaps.sort(key=lambda g: (-g["delta"], g["element_id"]))
    return gaps


def _ranked(gaps: list[dict], top_n: int) -> list[dict]:
    """Take the top N and stamp a 1-based rank onto each (copies, not aliases)."""
    top = [dict(g) for g in gaps[:top_n]]
    for rank, g in enumerate(top, start=1):
        g["rank"] = rank
    return top


def skill_gap(
    soc_current: str,
    soc_next: str,
    conn: sqlite3.Connection | None = None,
    top_n: int = 5,
) -> list[dict]:
    """
    Top trainable descriptors to develop when moving current -> next.

    One ranked list, whichever descriptor type wins. Only Skills + Knowledge
    descriptors exist in this DB, so Abilities are excluded by construction
    (they are never loaded).

    Each item: {rank, element_id, element_name, descriptor_type,
                lv_current, lv_next, delta}
    """
    conn, owned = _connect(conn)
    try:
        return _ranked(_all_gaps(soc_current, soc_next, conn), top_n)
    finally:
        if owned:
            conn.close()


def skill_gap_split(
    soc_current: str,
    soc_next: str,
    conn: sqlite3.Connection | None = None,
    top_n: int = 3,
) -> dict[str, list[dict]]:
    """
    Same deltas as skill_gap(), but split into Skills and Knowledge separately.

    Returns {'skills': [...], 'knowledge': [...]}, each independently ranked and
    capped at top_n.

    Why split: ranking one blended list by raw delta lets Knowledge sweep the
    board (Financial Analyst -> Financial Managers returns five Knowledge areas
    and zero Skills), which hides the trainable skills an HR reader is looking
    for. Splitting guarantees both kinds of recommendation surface.
    """
    conn, owned = _connect(conn)
    try:
        gaps = _all_gaps(soc_current, soc_next, conn)
        return {
            "skills": _ranked([g for g in gaps if g["descriptor_type"] == "skill"], top_n),
            "knowledge": _ranked([g for g in gaps if g["descriptor_type"] == "knowledge"], top_n),
        }
    finally:
        if owned:
            conn.close()
