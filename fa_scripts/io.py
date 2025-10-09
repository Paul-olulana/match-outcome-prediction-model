from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd

# ---------- logging ----------
logger = logging.getLogger("fa_io")
if not logger.handlers:
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

# ---------- constants ----------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"

EXPECTED_FILES = {
    "fbref_team_defense":        "fbref_team_defense.csv",
    "fbref_team_goal_shot_creation": "fbref_team_goal_shot_creation.csv",
    "fbref_team_match_stats":    "fbref_team_match_stats.csv",
    "fbref_team_misc":           "fbref_team_misc.csv",
    "fbref_team_passing":        "fbref_team_passing.csv",
    "fbref_team_possession":     "fbref_team_possession.csv",
    "fbref_team_schedule":       "fbref_team_schedule.csv",
    "fbref_team_shooting":       "fbref_team_shooting.csv",
    "understat_team_matches":    "understat_team_matches.csv",
}

# Columns that should be parsed as dates if present
DATE_GUESS_COLS = ("date", "match_date", "Date", "datetime")

def _resolve_paths() -> Dict[str, Path]:
    paths = {k: DATA_RAW / v for k, v in EXPECTED_FILES.items()}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        msg = "Missing raw files:\n" + "\n".join(missing)
        logger.error(msg)
        raise FileNotFoundError(msg)
    return paths

def _read_csv_safe(fp: Path) -> pd.DataFrame:
    """
    Robust CSV reader:
      - handles UTF-8 / BOM
      - trims whitespace in column names
      - parses obvious date columns
      - keeps numeric strings as-is; let later stages coerce
    """
    try:
        df = pd.read_csv(fp, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(fp, encoding="utf-8-sig")

    # normalize columns
    df.columns = [c.strip() for c in df.columns]

    # date parsing (only if column clearly a date; don't infer all)
    for col in df.columns:
        if col in DATE_GUESS_COLS:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)

    # strip string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()

    logger.info(f"Loaded {fp.name}: {df.shape[0]:,} rows × {df.shape[1]} cols")
    return df

def load_all() -> Dict[str, pd.DataFrame]:
    """
    Load all expected CSVs into memory as DataFrames.
    Keys match EXPECTED_FILES (e.g., 'fbref_team_passing', 'understat_team_matches').
    """
    paths = _resolve_paths()
    frames = {k: _read_csv_safe(fp) for k, fp in paths.items()}
    return frames

def quick_healthcheck(frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Small summary: rows, cols, non-null %, min/max date if any date col exists.
    """
    records = []
    for name, df in frames.items():
        row = {
            "name": name,
            "rows": len(df),
            "cols": df.shape[1],
            "non_null_pct": round(100 * (1 - df.isna().mean().mean()), 2),
        }
        date_cols = [c for c in df.columns if c in DATE_GUESS_COLS]
        if date_cols:
            d = df[date_cols[0]]
            if pd.api.types.is_datetime64_any_dtype(d):
                row["min_date"] = pd.to_datetime(d).min()
                row["max_date"] = pd.to_datetime(d).max()
        records.append(row)
    return pd.DataFrame(records).sort_values("name").reset_index(drop=True)

def save_checkpoint(frames: Dict[str, pd.DataFrame], suffix: str = "phase1_clean"):
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    out_dir = DATA_PROC / suffix
    out_dir.mkdir(exist_ok=True)
    for name, df in frames.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    logger.info(f"Checkpoint saved to {out_dir}")
