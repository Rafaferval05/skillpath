"""
SkillPath - single-command build + validate.

    python build.py                 # build data/onet.db (reuse zip if present), then validate
    python build.py --force-download # re-download the O*NET zip first

Definition of done: this one command builds onet.db from scratch and runs all five
validation suites - the Phase 0 math spine, the Phase 1 + 3 org handling and
adjacency engine, the deterministic half of the Phase 2 + 4 matching and UI layer,
the Phase 5 divergence flag and the Phase 6 team drift view - and every assertion
passes with readable output.
Exits non-zero if any suite fails. (No suite makes an API call, so no key is
needed here.)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import onet_loader  # noqa: E402
from tests import (  # noqa: E402
    validate,
    validate_divergence,
    validate_matching,
    validate_org,
    validate_team,
)


def main() -> int:
    force = "--force-download" in sys.argv

    print("#" * 70)
    print("# SkillPath - build O*NET spine + validate math, org handling, engine")
    print("#" * 70)

    onet_loader.build(force_download=force)

    # Every suite always runs, so one failure never hides another's result.
    print()
    rc_math = validate.main()

    print()
    rc_org = validate_org.main()

    print()
    rc_match = validate_matching.main()

    print()
    rc_div = validate_divergence.main()

    print()
    rc_team = validate_team.main()

    print()
    print("#" * 70)
    print(f"# Phase 0 math spine ....... {'PASS' if rc_math == 0 else 'FAIL'}")
    print(f"# Phase 1 + 3 org/engine ... {'PASS' if rc_org == 0 else 'FAIL'}")
    print(f"# Phase 2 + 4 match/UI ..... {'PASS' if rc_match == 0 else 'FAIL'}")
    print(f"# Phase 5 divergence ....... {'PASS' if rc_div == 0 else 'FAIL'}")
    print(f"# Phase 6 team drift ....... {'PASS' if rc_team == 0 else 'FAIL'}")
    print("#" * 70)
    return rc_math or rc_org or rc_match or rc_div or rc_team


if __name__ == "__main__":
    raise SystemExit(main())
