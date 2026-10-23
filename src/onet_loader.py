"""
Re-runnable O*NET loader for SkillPath Phase 0.

Downloads the O*NET 30.3 tab-delimited text database (if not already present)
and loads exactly the pieces the math spine needs into a local SQLite DB:

    - Occupation Data          -> occupation table (SOC code, title, description)
    - Essential Skills   (2.A) -\
    - Transferable Skills(2.B) -- } -> ratings table (descriptor IM/LV per role)
    - Knowledge          (2.C) -/

NOTE ON THE 30.3 RESTRUCTURE
----------------------------
The spec assumed a single "Skills.txt". O*NET 30.3 split the old ~35-descriptor
Skills content model into two files: "Essential Skills" (element IDs 2.A.*) and
"Transferable Skills" (2.B.*). Together they ARE the spec's ~35 skills, and both
still carry the Importance (IM) and Level (LV) scores the vector math relies on.
"Software Skills" is deliberately excluded: it has no IM/LV scores (only
Hot Technology / In Demand Y/N flags), so it does not fit the LV x IM model.

Resulting descriptor space: 35 skills + 33 knowledge = 68 dimensions, exactly
matching the spec's "~68 dimensions".

This script is idempotent: re-running it rebuilds every table from scratch.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import sys
import urllib.request
import zipfile
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ONET_VERSION = "30.3"
ZIP_URL = "https://www.onetcenter.org/dl_files/database/db_30_3_text.zip"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ZIP_PATH = DATA_DIR / "db_30_3_text.zip"
DB_PATH = DATA_DIR / "onet.db"

# Folder name inside the zip archive.
_ARCHIVE_PREFIX = "db_30_3_text"

# Descriptor source files -> descriptor_type used throughout the math layer.
# "skill" == Essential + Transferable (the spec's ~35 Skills).
DESCRIPTOR_SOURCES = {
    f"{_ARCHIVE_PREFIX}/Essential Skills.txt": "skill",
    f"{_ARCHIVE_PREFIX}/Transferable Skills.txt": "skill",
    f"{_ARCHIVE_PREFIX}/Knowledge.txt": "knowledge",
}
OCCUPATION_FILE = f"{_ARCHIVE_PREFIX}/Occupation Data.txt"

# Only these two scales matter for the vector / gap math.
KEEP_SCALES = {"IM", "LV"}

# Column headers, resolved by NAME rather than position so a reordered or renamed
# O*NET column fails loudly instead of silently feeding the wrong field to the math.
SOC_COL = "O*NET-SOC Code"
TITLE_COL = "Title"
DESCRIPTION_COL = "Description"
ELEMENT_ID_COL = "Element ID"
ELEMENT_NAME_COL = "Element Name"
SCALE_COL = "Scale ID"
VALUE_COL = "Data Value"
SUPPRESS_COL = "Recommend Suppress"

OCCUPATION_COLUMNS = (SOC_COL, TITLE_COL)
DESCRIPTOR_COLUMNS = (
    SOC_COL,
    ELEMENT_ID_COL,
    ELEMENT_NAME_COL,
    SCALE_COL,
    VALUE_COL,
    SUPPRESS_COL,
)


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def download_zip(force: bool = False) -> Path:
    """Download the O*NET text zip into ./data if it is not already there."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and not force:
        size = ZIP_PATH.stat().st_size
        print(f"[download] {ZIP_PATH.name} already present ({size:,} bytes) - skipping.")
        return ZIP_PATH

    print(f"[download] fetching O*NET {ONET_VERSION} text database ...")
    print(f"[download]   {ZIP_URL}")
    req = urllib.request.Request(ZIP_URL, headers={"User-Agent": "SkillPath-MVP/0.1"})
    with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as fh:
        fh.write(resp.read())
    print(f"[download] saved {ZIP_PATH} ({ZIP_PATH.stat().st_size:,} bytes)")
    return ZIP_PATH


# --------------------------------------------------------------------------- #
# Parsing helpers
# --------------------------------------------------------------------------- #

def _read_dict_rows(zf: zipfile.ZipFile, member: str, required: tuple[str, ...]):
    """
    Yield rows as {column_name: value} dicts from a tab-delimited O*NET member.

    Every column the loader consumes is addressed by header name. If O*NET ever
    reorders or renames one, this raises immediately instead of quietly reading
    the wrong field - which is how the "Recommend Suppress" filter came to be
    missed while columns were being read positionally.
    """
    with zf.open(member) as raw:
        # O*NET text files are UTF-8; be forgiving about stray bytes.
        text = io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
        reader = csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE)
        if reader.fieldnames is None:
            raise ValueError(f"'{member}' is empty - no header row.")
        missing = [c for c in required if c not in reader.fieldnames]
        if missing:
            raise KeyError(
                f"'{member}' is missing expected column(s) {missing}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            if row:
                yield row


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #

def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS occupation;
        DROP TABLE IF EXISTS ratings;
        DROP TABLE IF EXISTS descriptor;
        DROP TABLE IF EXISTS meta;

        CREATE TABLE occupation (
            soc_code    TEXT PRIMARY KEY,   -- e.g. '13-2051.00'
            title       TEXT NOT NULL,
            description TEXT
        );

        -- One row per (role, descriptor, scale). scale_id is 'IM' or 'LV'.
        CREATE TABLE ratings (
            soc_code        TEXT NOT NULL,
            element_id      TEXT NOT NULL,  -- e.g. '2.A.1.a'
            element_name    TEXT NOT NULL,  -- e.g. 'Reading Comprehension'
            descriptor_type TEXT NOT NULL,  -- 'skill' | 'knowledge'
            scale_id        TEXT NOT NULL,  -- 'IM' | 'LV'
            data_value      REAL NOT NULL
        );

        -- Canonical descriptor dimension (the 68 axes every vector spans).
        CREATE TABLE descriptor (
            element_id      TEXT PRIMARY KEY,
            element_name    TEXT NOT NULL,
            descriptor_type TEXT NOT NULL
        );

        CREATE TABLE meta (
            key   TEXT PRIMARY KEY,
            value TEXT
        );
        """
    )


def _create_indexes(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE INDEX idx_ratings_soc     ON ratings (soc_code);
        CREATE INDEX idx_ratings_element ON ratings (element_id);
        CREATE INDEX idx_ratings_lookup  ON ratings (soc_code, element_id, scale_id);
        """
    )


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #

def load(db_path: Path = DB_PATH, zip_path: Path = ZIP_PATH) -> dict:
    """(Re)build the SQLite database from the O*NET zip. Returns a summary dict."""
    if not zip_path.exists():
        raise FileNotFoundError(f"O*NET zip not found at {zip_path}; run download_zip() first.")

    conn = sqlite3.connect(db_path)
    try:
        _create_schema(conn)

        with zipfile.ZipFile(zip_path) as zf:
            members = set(zf.namelist())

            # --- Occupation Data -------------------------------------------- #
            if OCCUPATION_FILE not in members:
                raise KeyError(f"'{OCCUPATION_FILE}' missing from {zip_path.name}")
            occ_rows = [
                (r[SOC_COL], r[TITLE_COL], r.get(DESCRIPTION_COL))
                for r in _read_dict_rows(zf, OCCUPATION_FILE, OCCUPATION_COLUMNS)
            ]
            conn.executemany(
                "INSERT OR REPLACE INTO occupation (soc_code, title, description) VALUES (?,?,?)",
                occ_rows,
            )

            # --- Descriptor ratings (skills + knowledge) -------------------- #
            rating_rows: list[tuple] = []
            descriptors: dict[str, tuple[str, str]] = {}  # element_id -> (name, type)
            per_source: dict[str, int] = {}
            suppressed_per_source: dict[str, int] = {}

            for member, dtype in DESCRIPTOR_SOURCES.items():
                if member not in members:
                    raise KeyError(f"'{member}' missing from {zip_path.name}")
                count = 0
                suppressed = 0
                for r in _read_dict_rows(zf, member, DESCRIPTOR_COLUMNS):
                    scale_id = r[SCALE_COL]
                    if scale_id not in KEEP_SCALES:
                        continue
                    data_value = r[VALUE_COL]
                    if data_value == "" or data_value.lower() == "n/a":
                        continue
                    # O*NET marks estimates it considers too unreliable to publish.
                    # The spec drops them outright: with no LV/IM row to read, the
                    # descriptor simply contributes 0 for that occupation.
                    if (r[SUPPRESS_COL] or "").strip().upper() == "Y":
                        suppressed += 1
                        continue
                    elem_id, elem_name = r[ELEMENT_ID_COL], r[ELEMENT_NAME_COL]
                    rating_rows.append(
                        (r[SOC_COL], elem_id, elem_name, dtype, scale_id, float(data_value))
                    )
                    descriptors.setdefault(elem_id, (elem_name, dtype))
                    count += 1
                per_source[member] = count
                suppressed_per_source[member] = suppressed

            conn.executemany(
                "INSERT INTO ratings "
                "(soc_code, element_id, element_name, descriptor_type, scale_id, data_value) "
                "VALUES (?,?,?,?,?,?)",
                rating_rows,
            )
            conn.executemany(
                "INSERT INTO descriptor (element_id, element_name, descriptor_type) VALUES (?,?,?)",
                [(eid, name, dtype) for eid, (name, dtype) in descriptors.items()],
            )

        _create_indexes(conn)

        # --- Metadata ------------------------------------------------------- #
        n_occ = conn.execute("SELECT COUNT(*) FROM occupation").fetchone()[0]
        n_desc = conn.execute("SELECT COUNT(*) FROM descriptor").fetchone()[0]
        n_skill = conn.execute(
            "SELECT COUNT(*) FROM descriptor WHERE descriptor_type='skill'"
        ).fetchone()[0]
        n_know = conn.execute(
            "SELECT COUNT(*) FROM descriptor WHERE descriptor_type='knowledge'"
        ).fetchone()[0]
        n_ratings = conn.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]

        n_suppressed = sum(suppressed_per_source.values())

        meta = {
            "onet_version": ONET_VERSION,
            "source_zip": zip_path.name,
            "source_url": ZIP_URL,
            "n_occupations": str(n_occ),
            "n_descriptors": str(n_desc),
            "n_skill_descriptors": str(n_skill),
            "n_knowledge_descriptors": str(n_know),
            "n_rating_rows": str(n_ratings),
            # Rows O*NET flagged 'Recommend Suppress = Y' and this loader dropped.
            "n_suppressed_rows": str(n_suppressed),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?,?)", list(meta.items())
        )
        conn.commit()
    finally:
        conn.close()

    summary = {
        "db_path": str(db_path),
        "occupations": n_occ,
        "descriptors": n_desc,
        "skill_descriptors": n_skill,
        "knowledge_descriptors": n_know,
        "rating_rows": n_ratings,
        "suppressed_rows": n_suppressed,
        "per_source": per_source,
        "suppressed_per_source": suppressed_per_source,
    }
    return summary


def build(force_download: bool = False) -> dict:
    """Download (if needed) + load. This is the public one-call entrypoint."""
    download_zip(force=force_download)
    summary = load()
    print("[load] O*NET database built:")
    print(f"[load]   db: {summary['db_path']}")
    print(f"[load]   occupations:          {summary['occupations']:>6}")
    print(
        f"[load]   descriptors (dims):   {summary['descriptors']:>6} "
        f"({summary['skill_descriptors']} skill + {summary['knowledge_descriptors']} knowledge)"
    )
    print(f"[load]   rating rows (IM/LV):  {summary['rating_rows']:>6}")
    print(f"[load]   suppressed (dropped): {summary['suppressed_rows']:>6}")
    for member, n in summary["suppressed_per_source"].items():
        print(f"[load]     {member.split('/')[-1]:<26} {n:>5}")
    return summary


if __name__ == "__main__":
    force = "--force-download" in sys.argv
    build(force_download=force)
