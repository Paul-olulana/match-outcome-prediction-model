# fa_scripts/name_map.py
from __future__ import annotations
from typing import Dict, Iterable
import pandas as pd
import unicodedata

# --- Choose ONE canonical spelling per team (prefer what appears in fbref_team_schedule) ---
# Keep this list short and explicit; add rows only when you see a mismatch in the report.
TEAM_MAP: Dict[str, str] = {
    # Paris SG variants
    "Paris S-G": "Paris Saint-Germain",
    "Paris Saint Germain": "Paris Saint-Germain",
    # Saint-Etienne variants
    "Saint-Étienne": "Saint-Etienne",
    "AS Saint-Étienne": "Saint-Etienne",
    "AS Saint-Etienne": "Saint-Etienne",
    # Nimes variants
    "Nîmes": "Nimes",

    # (Add any others you find…)
}

# Optionally also accept diacritic-stripped versions mapping to canonical
# (kept explicit—no fuzzy fallbacks)
TEAM_MAP.update({
    "Nimes Olympique": "Nimes",  # if it appears
})

CANONICAL_COLS = {"team", "squad", "home", "away", "opponent", "h_team", "a_team", "Team"}

def normalize_team_value(v: str) -> str:
    """Return mapped/canonical team string if in TEAM_MAP, else original."""
    if pd.isna(v):
        return v
    v = str(v).strip()
    if v in TEAM_MAP:
        return TEAM_MAP[v]
    # Try a diacritics-stripped key (still explicit—only if present in map)
    v_ascii = unicodedata.normalize("NFKD", v).encode("ascii", "ignore").decode("ascii")
    return TEAM_MAP.get(v_ascii, v)

def apply_team_map(df: pd.DataFrame, cols: Iterable[str] | None = None) -> pd.DataFrame:
    """Apply TEAM_MAP to selected columns (if present)."""
    out = df.copy()
    cols = list(cols) if cols else [c for c in out.columns if c in CANONICAL_COLS]
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(normalize_team_value)
    return out

def assert_all_teams_mapped(df: pd.DataFrame, ref_set: set[str], cols: Iterable[str] | None = None):
    """Raise if df contains teams not present in ref_set (after mapping)."""
    cols = list(cols) if cols else [c for c in df.columns if c in CANONICAL_COLS]
    unknown = set()
    for c in cols:
        if c in df.columns:
            unknown |= set(out for out in df[c].dropna().unique() if out not in ref_set)
    if unknown:
        raise AssertionError(
            "Found unmapped team(s) not in reference set after applying TEAM_MAP:\n"
            + "\n".join(sorted(unknown))
        )
