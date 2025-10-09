# fa_scripts/preprocess.py
from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd

# ----------------------------
# 0) Utilities
# ----------------------------
def _to_datetime(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce")
    return s

def _strip_percent_to_float(s: pd.Series) -> pd.Series:
    if s.dtype == "O":
        s = s.str.replace("%", "", regex=False).str.replace(",", "", regex=False)
    return pd.to_numeric(s, errors="coerce")

def clean_numeric_percent(df: pd.DataFrame, percent_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in percent_cols:
        if c in df.columns:
            df[c] = _strip_percent_to_float(df[c]) / 100.0
    return df

def ensure_types(df: pd.DataFrame, dt_cols=("date",), cat_cols=("team","opponent","season")) -> pd.DataFrame:
    df = df.copy()
    for c in dt_cols:
        if c in df.columns:
            df[c] = _to_datetime(df[c])
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df
def infer_season_from_date(date: pd.Timestamp, cutoff_month: int = 7) -> str:
    if pd.isna(date): 
        return pd.NA
    y = int(date.year)
    if int(date.month) >= cutoff_month:
        return str(y)
    return str(y - 1)

def ensure_season_column(df: pd.DataFrame, date_col: str = "date", cutoff_month: int = 7) -> pd.DataFrame:
    out = df.copy()
    if "season" not in out.columns and date_col in out.columns:
        out["season"] = out[date_col].apply(lambda d: infer_season_from_date(pd.to_datetime(d, errors="coerce"), cutoff_month))
    out["season"] = out["season"].astype("string")
    return out

# ----------------------------
# 1) Create team_rows if needed
# ----------------------------
def standardize_schedule(schedule_like: pd.DataFrame) -> pd.DataFrame:
    """Standardize a schedule-like df into [date, season, home, away, home_goals, away_goals]."""
    df = schedule_like.copy()
    df = df.rename(columns={c: c.lower() for c in df.columns})

    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    date_c = pick("date", "match_date", "datetime")
    home_c = pick("home", "home_team", "h_team")
    away_c = pick("away", "away_team", "a_team")
    hg_c = pick("home_goals", "fthg", "hg", "home_score", "score_home")
    ag_c = pick("away_goals", "ftha", "ag", "away_score", "score_away")
    season_c = pick("season", "year", "season_id")

    need = [date_c, home_c, away_c, hg_c, ag_c]
    if any(x is None for x in need):
        raise ValueError("Schedule is missing required columns.")

    cols = [date_c, home_c, away_c, hg_c, ag_c] + ([season_c] if season_c else [])
    out = df[cols].copy()
    out.columns = ["date", "home", "away", "home_goals", "away_goals"] + (["season"] if season_c else [])
    out["date"] = _to_datetime(out["date"])

    if season_c:
        out["season"] = pd.to_numeric(out["season"], errors="coerce")
    else:
        out["season"] = out["date"].dt.year

    out["season"] = out["season"].astype("Int64").astype("string")
    return out

def explode_to_team_rows(schedule_std: pd.DataFrame) -> pd.DataFrame:
    """Two rows per match: one for home team, one for away."""
    a = schedule_std.copy()
    home = a.assign(team=a["home"], opponent=a["away"], is_home=1,
                    gf=a["home_goals"], ga=a["away_goals"])
    away = a.assign(team=a["away"], opponent=a["home"], is_home=0,
                    gf=a["away_goals"], ga=a["home_goals"])
    long = pd.concat([home, away], ignore_index=True)
    long["date"] = _to_datetime(long["date"])
    long["result"] = np.where(long["gf"]>long["ga"], "win",
                       np.where(long["gf"]<long["ga"], "loss", "draw"))
    long["match_id"] = long["date"].dt.strftime("%Y%m%d") + "_" + long["home"] + "_" + long["away"]
    return long

# ----------------------------
# 2) Rolling (no leakage)
# ----------------------------
def rolling_mean_strict(s: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return s.shift(1).rolling(window=window, min_periods=min_periods).mean()

def rolling_sum_strict(s: pd.Series, window: int, min_periods: int = 1) -> pd.Series:
    return s.shift(1).rolling(window=window, min_periods=min_periods).sum()

def exp_weighted_strict(s: pd.Series, alpha: float = 0.8, adjust: bool = False) -> pd.Series:
    return s.shift(1).ewm(alpha=alpha, adjust=adjust).mean()

def build_team_form(
    team_rows: pd.DataFrame,
    metrics_mean: List[str],
    metrics_sum: Optional[List[str]] = None,
    windows: Tuple[int, ...] = (3,5,10),
    exp_alpha: Optional[float] = 0.8,
    per90_cols: Optional[Dict[str, str]] = None  # {"cards":"minutes"} if available
) -> pd.DataFrame:
    """
    Compute rolling features for each team (strictly past). Returns a wide df with new columns appended.
    metrics_mean: columns to rolling-mean (e.g., xg, poss, pass_acc)
    metrics_sum : columns to rolling-sum (e.g., goals, cards)
    """
    df = team_rows.sort_values(["team","date"]).copy()
    g = df.groupby("team", group_keys=False)

    def add_rolls(col, fn, suffix):
        cname = f"{col}_{suffix}"
        df[cname] = g[col].apply(fn)
        return cname

    created = []
    if metrics_mean:
        for col in metrics_mean:
            for w in windows:
                created.append(add_rolls(col, lambda s, w=w: rolling_mean_strict(s, w), f"l{w}"))
            if exp_alpha is not None:
                created.append(add_rolls(col, lambda s: exp_weighted_strict(s, exp_alpha), f"exp{int(100*exp_alpha)}"))

    if metrics_sum:
        for col in metrics_sum:
            for w in windows:
                created.append(add_rolls(col, lambda s, w=w: rolling_sum_strict(s, w), f"sum_l{w}"))

    # optional per-90 conversion if minutes exist
    if per90_cols:
        for raw_col, minutes_col in per90_cols.items():
            if raw_col in df.columns and minutes_col in df.columns:
                rate_col = f"{raw_col}_per90"
                df[rate_col] = (df[raw_col] / (df[minutes_col].replace(0, np.nan) / 90.0)).replace([np.inf,-np.inf], np.nan)
                created.append(rate_col)

    return df, created

# ----------------------------
# 3) Pivot to match-level (home/away side-by-side)
# ----------------------------
def _pivot_home_away(team_rows_with_form: pd.DataFrame, keep_cols: List[str]) -> pd.DataFrame:
    if "season" not in team_rows_with_form.columns or team_rows_with_form["season"].isna().all():
        team_rows_with_form = team_rows_with_form.copy()
        team_rows_with_form["season"] = team_rows_with_form["date"].dt.year.astype("Int64").astype("string")

    keep = ["match_id","date","season","home","away","team","opponent","is_home","result","gf","ga"] + keep_cols
    df = team_rows_with_form[keep].copy()

    home = df[df["is_home"]==1].copy()
    away = df[df["is_home"]==0].copy()

    def add_prefix(d, pfx):
        d = d.drop(columns=["team","opponent","is_home","result"])
        d = d.rename(columns={c: f"{pfx}_{c}" for c in d.columns if c not in ["match_id","date","season","home","away","gf","ga"]})
        d = d.rename(columns={"gf": f"{pfx}_goals", "ga": f"{pfx}_conceded"})
        return d

    H = add_prefix(home, "home")
    A = add_prefix(away, "away")

    # merge on match keys
    out = pd.merge(
        H, A,
        on=["match_id","date","season","home","away"],
        how="inner",
        suffixes=("","")
    )

    # final label for home team perspective
    # (1 = home win, 0 = draw, -1 = away win)
    out["result"] = np.where(out["home_goals"]>out["away_goals"], 1,
                      np.where(out["home_goals"]<out["away_goals"], -1, 0))
    return out

# ----------------------------
# 4) Feature diffs & housekeeping
# ----------------------------
def add_differentials(df: pd.DataFrame, pairs: List[Tuple[str,str]], suffix="_diff") -> pd.DataFrame:
    df = df.copy()
    for left, right in pairs:
        if left in df.columns and right in df.columns:
            df[left + suffix] = df[left] - df[right]
    return df

def train_val_test_split_time(
    df: pd.DataFrame,
    train_seasons: List[str],
    val_seasons: List[str],
    test_seasons: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tr = df[df["season"].isin(train_seasons)].copy()
    va = df[df["season"].isin(val_seasons)].copy()
    te = df[df["season"].isin(test_seasons)].copy()
    return tr, va, te

# ----------------------------
# 5) High-level pipeline
# ----------------------------
def build_match_level_dataset(
    team_rows: pd.DataFrame,
    percent_cols: Optional[List[str]] = None,
    mean_metrics: Optional[List[str]] = None,
    sum_metrics: Optional[List[str]] = None,
    windows: Tuple[int,...] = (3,5,10),
    exp_alpha: float = 0.8,
    keep_current_cols: Optional[List[str]] = None,
    add_diff_for: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    1) clean types & percents
    2) compute rolling features on team_rows (strict past)
    3) pivot to match-level (home/away)
    4) add differentials (home - away)
    """
    df = ensure_types(team_rows, dt_cols=("date",), cat_cols=("team","opponent","season"))
    if "season" not in df.columns or df["season"].isna().all():
        df["season"] = df["date"].dt.year.astype("Int64").astype("string")
    if percent_cols:
        df = clean_numeric_percent(df, percent_cols)

    # defaults based on your Phase 1 note; robust if xA not present
    if mean_metrics is None:
        mean_metrics = [c for c in ["xg","npxg","poss","pass_acc"] if c in df.columns]
    if sum_metrics is None:
        sum_metrics = [c for c in ["gf","ga","yellow_cards","red_cards","clearances"] if c in df.columns]

    df_roll, created_cols = build_team_form(
        df,
        metrics_mean=mean_metrics,
        metrics_sum=sum_metrics,
        windows=windows,
        exp_alpha=exp_alpha
    )

    # columns to carry into pivot (current-match stats are allowed for reporting,
    # but be careful not to train on same-match stats—use *_l* & *_exp* for model)
    if keep_current_cols is None:
        keep_current_cols = [c for c in ["xg","npxg","poss","pass_acc","yellow_cards","red_cards","clearances"] if c in df.columns]

    keep_cols = keep_current_cols + created_cols

    match_level = _pivot_home_away(df_roll, keep_cols=keep_cols)

    # add differentials for recent-form features and core stats
    if add_diff_for is None:
        add_diff_for = ["xg","npxg","poss","pass_acc"]
        # include roll columns as well:
        add_diff_for += [c for c in created_cols if any(k in c for k in ["xg","npxg","poss","pass_acc"])]

    pairs = []
    for base in add_diff_for:
        hl, al = f"home_{base}", f"away_{base}"
        if hl in match_level.columns and al in match_level.columns:
            pairs.append((hl, al))
    match_level = add_differentials(match_level, pairs, suffix="_diff")

    return match_level
