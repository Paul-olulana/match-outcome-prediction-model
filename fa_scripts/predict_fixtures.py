#!/usr/bin/env python
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fa_scripts.infer import (
    load_model_bundle,
    build_features_for_fixtures,
    apply_draw_override,
    symbol_map,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures_csv",
        required=True,
        help="CSV with upcoming fixtures (columns: date, home, away, optional season)",
    )
    parser.add_argument(
        "--history_csv",
        required=True,
        help="Historical match level file (phase2 match_level_dataset.csv)",
    )
    parser.add_argument(
        "--team_rows_csv",
        default=None,
        help="Optional Phase 2 team_rows snapshot for richer rolling features.",
    )
    parser.add_argument(
        "--models_dir",
        default="models",
        help="Directory containing champion_calibrated.joblib/model_card.json",
    )
    parser.add_argument(
        "--out_csv",
        default="reports/dashboard/predictions_upcoming.csv",
        help="Where to write the predictions CSV for Power BI",
    )
    args = parser.parse_args()

    fixtures = pd.read_csv(args.fixtures_csv, parse_dates=["date"])
    history = pd.read_csv(args.history_csv, parse_dates=["date"])
    team_rows_snapshot = (
        pd.read_csv(args.team_rows_csv, parse_dates=["date"])
        if args.team_rows_csv
        else None
    )

    bundle = load_model_bundle(Path(args.models_dir))
    model = bundle["model"]
    tau_draw = bundle["tau_draw"]
    feature_cols = bundle.get("feature_cols")
    label_order = bundle["label_order"]
    label_symbols = bundle["label_symbols"]

    features = build_features_for_fixtures(
        fixtures,
        history_match_level=history,
        team_rows_snapshot=team_rows_snapshot,
    )

    if feature_cols:
        missing = [c for c in feature_cols if c not in features.columns]
        if missing:
            # Create NaNs for missing columns (imputer inside pipeline will handle them)
            for col in missing:
                features[col] = np.nan
        features = features[feature_cols]

    proba = model.predict_proba(features)
    preds, parts = apply_draw_override(proba, bundle["label_order"], tau_draw)


    output = fixtures.copy()
    output["p_home_win"] = parts["p_home_win"]
    output["p_draw"] = parts["p_draw"]
    output["p_away_win"] = parts["p_away_win"]
    output["confidence"] = parts["confidence"]
    output["pred_label"] = preds
    output["pred_symbol"] = symbol_map(preds)
    output["draw_override_tau"] = tau_draw

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(out_path, index=False)

    print(f"Saved predictions to {out_path}")


if __name__ == "__main__":
    main()
