from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import json
import warnings

import numpy as np
import pandas as pd
import joblib

from .preprocess import (
    ensure_types,
    ensure_season_column,
    build_team_form,
    add_differentials,
)

# ------------------------------------------------------------------
# 0) Model bundle I/O
# ------------------------------------------------------------------
def load_model_bundle(models_dir: Path) -> Dict[str, Any]:
    """
    champion_calibrated.joblib must contain:
      - model            (CalibratedClassifierCV)
      - tau_draw         (float)
      - label_order      (list of original labels, e.g. [-1, 0, 1])
      - encoded_classes  (list; model.classes_, e.g. [0, 1, 2])
      - feature_cols     (optional)
      - label_symbols    (optional dict {-1:"A",0:"D",1:"H"})
    """
    bundle_path = models_dir / "champion_calibrated.joblib"
    bundle: Dict[str, Any] = joblib.load(bundle_path)

    required = {"model", "tau_draw", "label_order", "encoded_classes"}
    missing = required - set(bundle)
    if missing:
        raise KeyError(f"Model bundle missing required keys: {missing}")

    with open(models_dir / "model_card.json", "r") as f:
        card = json.load(f)

    bundle["model_card"] = card
    bundle["feature_cols"] = bundle.get("feature_cols", card.get("features_used"))
    bundle["label_symbols"] = bundle.get("label_symbols", {-1: "A", 0: "D", 1: "H"})
    return bundle


# ------------------------------------------------------------------
# 1) Feature build for LIVE fixtures
# ------------------------------------------------------------------
def build_features_for_fixtures(
    fixtures: pd.DataFrame,
    history_match_level: pd.DataFrame,
    team_rows_snapshot: Optional[pd.DataFrame] = None,
    mean_metrics: Optional[List[str]] = None,
    sum_metrics: Optional[List[str]] = None,
    windows: Tuple[int, ...] = (3, 5, 10),
    exp_alpha: float = 0.8,
) -> pd.DataFrame:
    fixtures = fixtures.copy()
    fixtures["date"] = pd.to_datetime(fixtures["date"], errors="coerce")
    fixtures = ensure_season_column(fixtures)

    # --- historical team rows -----------------------------------------------------
    if team_rows_snapshot is not None:
        team_rows_hist = team_rows_snapshot.copy()
        team_rows_hist = ensure_types(
            team_rows_hist,
            dt_cols=("date",),
            cat_cols=("team", "opponent", "season"),
        )
        team_rows_hist = ensure_season_column(team_rows_hist)
    else:
        warnings.warn(
            "team_rows_snapshot not supplied; falling back to goal-based rolling stats. "
            "Provide the Phase 2 team_rows snapshot for xG/possession features.",
            RuntimeWarning,
        )
        df_hist = history_match_level.copy()
        df_hist["date"] = pd.to_datetime(df_hist["date"], errors="coerce")
        for col in ["home", "away", "season"]:
            if col in df_hist.columns:
                df_hist[col] = df_hist[col].astype("string")

        home_rows = df_hist.rename(
            columns={"home": "team", "away": "opponent", "home_goals": "gf", "away_goals": "ga"}
        ).assign(is_home=1)
        away_rows = df_hist.rename(
            columns={"away": "team", "home": "opponent", "away_goals": "gf", "home_goals": "ga"}
        ).assign(is_home=0)

        base_cols = ["date", "season", "team", "opponent", "is_home", "gf", "ga"]
        team_rows_hist = pd.concat(
            [home_rows[base_cols], away_rows[base_cols]], ignore_index=True
        )
        team_rows_hist = ensure_types(
            team_rows_hist,
            dt_cols=("date",),
            cat_cols=("team", "opponent", "season"),
        )
        team_rows_hist = ensure_season_column(team_rows_hist)

    # --- rolling form -------------------------------------------------------------
    if mean_metrics is None:
        mean_metrics = [c for c in ["xg", "npxg", "poss", "pass_acc"] if c in team_rows_hist.columns]
    if sum_metrics is None:
        sum_metrics = [
            c for c in ["gf", "ga", "yellow_cards", "red_cards", "clearances"]
            if c in team_rows_hist.columns
        ]

    team_rows_form, created_cols = build_team_form(
        team_rows_hist,
        metrics_mean=mean_metrics,
        metrics_sum=sum_metrics,
        windows=windows,
        exp_alpha=exp_alpha,
    )

    for col in ["xg", "npxg", "poss", "pass_acc", "yellow_cards", "red_cards", "clearances"]:
        if col not in team_rows_form.columns:
            team_rows_form[col] = np.nan

    def last_state_before(team: str, when: pd.Timestamp) -> Optional[pd.Series]:
        sub = team_rows_form[(team_rows_form["team"] == team) & (team_rows_form["date"] < when)]
        if sub.empty:
            return None
        return sub.sort_values("date").iloc[-1]

    rows: List[Dict[str, Any]] = []
    for _, row in fixtures.iterrows():
        home_state = last_state_before(row["home"], row["date"])
        away_state = last_state_before(row["away"], row["date"])

        feature_row: Dict[str, Any] = {
            "match_id": row["date"].strftime("%Y%m%d") + f"_{row['home']}_{row['away']}",
            "date": row["date"],
            "season": row.get("season"),
            "home": row["home"],
            "away": row["away"],
        }
        if home_state is not None:
            for col in created_cols:
                feature_row[f"home_{col}"] = home_state[col]
        if away_state is not None:
            for col in created_cols:
                feature_row[f"away_{col}"] = away_state[col]
        rows.append(feature_row)

    features = pd.DataFrame(rows)

    diff_pairs = []
    for col in created_cols:
        hl, al = f"home_{col}", f"away_{col}"
        if hl in features.columns and al in features.columns:
            diff_pairs.append((hl, al))
    features = add_differentials(features, diff_pairs, suffix="_diff")

    return features


# ------------------------------------------------------------------
# 2) Post-processing: predictions & symbols
# ------------------------------------------------------------------
def apply_draw_override(prob: np.ndarray, label_order: List[int], tau_draw: float):
    """
    prob: (n, K) calibrated probabilities aligned with label_order.
    label_order: list of original labels (e.g. [-1, 0, 1]) in column order.
    """
    label_order = list(label_order)
    home_idx = label_order.index(1)
    draw_idx = label_order.index(0)
    away_idx = label_order.index(-1)

    top_idx = prob.argmax(axis=1)
    confidence = prob.max(axis=1)
    predictions = np.array(label_order)[top_idx]
    predictions[confidence < tau_draw] = 0  # override to draw

    return predictions, {
        "p_home_win": prob[:, home_idx],
        "p_draw": prob[:, draw_idx],
        "p_away_win": prob[:, away_idx],
        "confidence": confidence,
    }


def symbol_map(y: np.ndarray) -> np.ndarray:
    return np.where(y == 1, "H", np.where(y == 0, "D", "A"))
