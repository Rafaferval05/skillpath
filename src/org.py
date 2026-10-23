"""
SkillPath Phase 1 - org handling (pure, deterministic, NO LLM).

Loads an org-structure JSON (spec section 5), validates it against both its own
internal consistency and the O*NET data actually present in the DB, and walks the
reporting line.

Public API
----------
    load_org(path)                 -> Org
    validate_org(org, conn)        -> list[Issue]
    org_path(org, role_id)         -> list[Role]   (this role -> ... -> root)
    senior_roles(org, role)        -> list[Role]   (every role at a higher level)
    next_level_roles(org, role)    -> list[Role]   (roles at the NEAREST level above)

org_path is the *org* ladder - where the org chart says people go. The engine
(src/engine.py) computes the *skill* ladder. Phase 5's divergence flag is the
difference between the two, which is why both live behind clean functions.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from src.vectors import is_rated, occupation_title

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ORG_PATH = PROJECT_ROOT / "orgs" / "sample_finance.json"

# Spec section 5: 1 = entry, 2 = analyst, 3 = senior, 4 = manager, 5 = director.
MIN_LEVEL, MAX_LEVEL = 1, 5

REQUIRED_ROLE_KEYS = ("role_id", "title", "level")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Role:
    role_id: str
    title: str
    level: int
    reports_to: str | None = None
    headcount: int | None = None
    onet_soc: str | None = None
    description: str | None = None

    def label(self) -> str:
        return f"{self.title} ({self.onet_soc})" if self.onet_soc else self.title


@dataclass
class Org:
    company: str
    function: str
    roles: list[Role]
    by_id: dict[str, Role] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # Last one wins on duplicate ids; validate_org reports the duplicate as an error.
        self.by_id = {r.role_id: r for r in self.roles}

    def get(self, role_id: str) -> Role:
        try:
            return self.by_id[role_id]
        except KeyError:
            raise KeyError(
                f"No role '{role_id}' in this org. Known: {', '.join(sorted(self.by_id))}"
            ) from None

    def levels(self) -> list[int]:
        """Sorted, de-duplicated levels that actually have roles in them."""
        return sorted({r.level for r in self.roles})


@dataclass(frozen=True)
class Issue:
    severity: str   # 'error' | 'warning'
    role_id: str | None
    message: str

    def __str__(self) -> str:
        where = f"[{self.role_id}] " if self.role_id else ""
        return f"{self.severity.upper():<7} {where}{self.message}"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_org(path: str | Path = SAMPLE_ORG_PATH) -> Org:
    """
    Parse an org JSON into an Org. Raises on structurally unusable input.

    Semantic problems (bad reporting lines, unrated SOCs, ...) are NOT raised
    here - they are reported by validate_org so the caller can show them all at
    once rather than failing on the first.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Org file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)

    if not isinstance(raw, dict) or "roles" not in raw:
        raise ValueError(f"{path.name}: expected a JSON object with a 'roles' list.")
    if not isinstance(raw["roles"], list) or not raw["roles"]:
        raise ValueError(f"{path.name}: 'roles' must be a non-empty list.")

    roles: list[Role] = []
    for i, r in enumerate(raw["roles"]):
        missing = [k for k in REQUIRED_ROLE_KEYS if r.get(k) in (None, "")]
        if missing:
            raise ValueError(
                f"{path.name}: role #{i} is missing required key(s): {', '.join(missing)}"
            )
        roles.append(
            Role(
                role_id=str(r["role_id"]),
                title=str(r["title"]),
                level=int(r["level"]),
                reports_to=r.get("reports_to") or None,
                headcount=r.get("headcount"),
                onet_soc=r.get("onet_soc") or None,
                description=r.get("description") or None,
            )
        )

    return Org(
        company=str(raw.get("company", path.stem)),
        function=str(raw.get("function", "Unspecified")),
        roles=roles,
    )


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def validate_org(org: Org, conn: sqlite3.Connection | None = None) -> list[Issue]:
    """
    Check the org against itself and against the O*NET DB.

    Errors   - the engine cannot be trusted to run: duplicate ids, dangling or
               cyclic reporting lines, no single root, out-of-range levels, and
               SOC codes that are unknown or (the O*NET 30.3 gotcha) UNRATED.
    Warnings - suspicious but survivable: a manager not senior to their report.
    """
    issues: list[Issue] = []

    # --- ids ---------------------------------------------------------------- #
    seen: set[str] = set()
    for r in org.roles:
        if r.role_id in seen:
            issues.append(Issue("error", r.role_id, "duplicate role_id"))
        seen.add(r.role_id)

    # --- levels ------------------------------------------------------------- #
    for r in org.roles:
        if not MIN_LEVEL <= r.level <= MAX_LEVEL:
            issues.append(
                Issue("error", r.role_id, f"level {r.level} outside {MIN_LEVEL}-{MAX_LEVEL}")
            )

    # --- reporting lines ---------------------------------------------------- #
    roots = [r for r in org.roles if r.reports_to is None]
    if not roots:
        issues.append(Issue("error", None, "no root role (every role has a reports_to)"))
    elif len(roots) > 1:
        issues.append(
            Issue(
                "error",
                None,
                f"{len(roots)} root roles, expected exactly 1: "
                f"{', '.join(r.role_id for r in roots)}",
            )
        )

    for r in org.roles:
        if r.reports_to is None:
            continue
        if r.reports_to not in org.by_id:
            issues.append(
                Issue("error", r.role_id, f"reports_to '{r.reports_to}' is not a known role_id")
            )
            continue
        mgr = org.by_id[r.reports_to]
        if mgr.level <= r.level:
            issues.append(
                Issue(
                    "warning",
                    r.role_id,
                    f"manager '{mgr.role_id}' is level {mgr.level}, not above this role's "
                    f"level {r.level}",
                )
            )

    # Cycles: walk each role upward with a visited set; report each cycle once.
    reported_cycles: set[frozenset[str]] = set()
    for r in org.roles:
        chain: list[str] = []
        cursor = r
        while cursor.reports_to in org.by_id:
            chain.append(cursor.role_id)
            cursor = org.by_id[cursor.reports_to]
            if cursor.role_id in chain:
                key = frozenset(chain + [cursor.role_id])
                if key not in reported_cycles:
                    reported_cycles.add(key)
                    issues.append(
                        Issue(
                            "error",
                            r.role_id,
                            "reporting cycle: " + " -> ".join(chain + [cursor.role_id]),
                        )
                    )
                break

    # --- O*NET codes -------------------------------------------------------- #
    for r in org.roles:
        if not r.onet_soc:
            issues.append(
                Issue("error", r.role_id, "no onet_soc (Phase 2 matching would populate this)")
            )
            continue
        if not is_rated(r.onet_soc, conn):
            title = occupation_title(r.onet_soc, conn)
            known = title != r.onet_soc
            detail = (
                f"'{title}' exists in O*NET but has NO skill/knowledge ratings"
                if known
                else "not found in the O*NET occupation table"
            )
            issues.append(
                Issue("error", r.role_id, f"SOC {r.onet_soc}: {detail} - no vector can be built")
            )

    return issues


def errors(issues: list[Issue]) -> list[Issue]:
    """Just the blocking issues - the engine should not run while any remain."""
    return [i for i in issues if i.severity == "error"]


# --------------------------------------------------------------------------- #
# Walking the org chart
# --------------------------------------------------------------------------- #

def org_path(org: Org, role_id: str) -> list[Role]:
    """
    The naive org-path walk: this role, then its manager, up to the root.

    Always includes the starting role as element 0. Stops defensively if the
    reporting line cycles (validate_org reports that as an error).
    """
    path: list[Role] = []
    seen: set[str] = set()
    cursor: Role | None = org.get(role_id)
    while cursor is not None and cursor.role_id not in seen:
        path.append(cursor)
        seen.add(cursor.role_id)
        cursor = org.by_id.get(cursor.reports_to) if cursor.reports_to else None
    return path


def senior_roles(org: Org, role: Role) -> list[Role]:
    """Every role at a strictly higher level, ordered by level then role_id."""
    return sorted(
        (r for r in org.roles if r.level > role.level),
        key=lambda r: (r.level, r.role_id),
    )


def next_level_roles(org: Org, role: Role) -> list[Role]:
    """
    Roles at the NEAREST occupied level above this role.

    Nearest-occupied rather than level+1 so a gap in the org's level numbering
    (say no level 3 at all) does not dead-end the ladder.
    """
    above = [lv for lv in org.levels() if lv > role.level]
    if not above:
        return []
    target = above[0]
    return sorted((r for r in org.roles if r.level == target), key=lambda r: r.role_id)
