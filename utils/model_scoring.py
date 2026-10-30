import os
import json
import joblib
import pandas as pd
from datetime import datetime, timezone

import utils.model_promotion

PRED_COLS = ["loan_id", "Customer_ID", "snapshot_date", "model_version", "arm", "score", "threshold", "prediction", "scored_at"]


def score_with_model(df, model_store_directory, version):
    # the model's own metadata says which columns it was trained on, so scoring cannot drift from training
    meta = utils.model_promotion.load_metadata(model_store_directory, version)
    pipe = joblib.load(model_store_directory + version + "/model.pkl")
    X = df[meta["num_cols"] + meta["cat_cols"]]
    score = pipe.predict_proba(X)[:, 1]
    return score, meta["threshold"]


def run_scoring(snapshot_date_str, feature_store_directory, model_store_directory, predictions_directory):
    # snapshot_date_str is the run month = application month of the loans being scored
    suffix = snapshot_date_str.replace('-', '_') + ".parquet"
    feature_path = feature_store_directory + "gold_feature_store_" + suffix

    # 1. Skip cleanly when there is nothing to score or no model yet
    if not os.path.exists(feature_path):
        print(snapshot_date_str, "skipped - no feature partition (no new applications this month)")
        return None
    pointer = utils.model_promotion.read_champion(model_store_directory)
    if pointer is None:
        print(snapshot_date_str, "skipped - no champion in the model store yet")
        return None

    df = pd.read_parquet(feature_path)
    print("loaded", len(df), "loans from", feature_path)

    # 2. Score with both arms. Same model twice is kept as two rows so every loan has one row per arm
    arms = {"champion": pointer["champion"], "control": pointer["control"]}
    scored_at = datetime.now(timezone.utc).isoformat()
    frames = []
    for arm, version in arms.items():
        score, threshold = score_with_model(df, model_store_directory, version)
        out = df[["loan_id", "Customer_ID", "snapshot_date"]].copy()
        out["model_version"] = version
        out["arm"] = arm
        out["score"] = score
        out["threshold"] = threshold
        out["prediction"] = (score >= threshold).astype(int)
        out["scored_at"] = scored_at
        frames.append(out[PRED_COLS])
        print(arm, version, "| flagged", round(out["prediction"].mean(), 3), "| mean score", round(float(score.mean()), 3))
    result = pd.concat(frames, ignore_index=True)

    # 3. One partition per application month; rerunning the same month overwrites only that month 
    os.makedirs(predictions_directory, exist_ok=True)
    out_path = predictions_directory + "gold_predictions_" + suffix
    result.to_parquet(out_path, index=False)
    print("saved to:", out_path, "| rows", len(result))
    return result