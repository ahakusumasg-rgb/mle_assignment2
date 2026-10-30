import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer
from xgboost import XGBClassifier

ID_COLS = ["loan_id", "Customer_ID", "snapshot_date"]
FIRST_APPLICATION_MONTH = "2023-01-01"
MOB = 6
MAX_WINDOW_MONTHS = 12


def get_training_window(run_date_str, mob=MOB, max_months=MAX_WINDOW_MONTHS, first_month=FIRST_APPLICATION_MONTH):
    # at run D only loans applied up to D - mob have a known label; window grows, then rolls at max_months
    run_date = datetime.strptime(run_date_str, "%Y-%m-%d")
    end = run_date - relativedelta(months=mob)
    start = max(datetime.strptime(first_month, "%Y-%m-%d"), end - relativedelta(months=max_months - 1))
    months = []
    current = start
    while current <= end:
        months.append(current.strftime("%Y-%m-%d"))
        current += relativedelta(months=1)
    return months


def load_training_data(window_months, feature_store_directory, label_store_directory, mob=MOB):
    # features are keyed by application month A, labels by label month L = A + mob; join on loan_id
    frames = []
    for a in window_months:
        label_date = (datetime.strptime(a, "%Y-%m-%d") + relativedelta(months=mob)).strftime("%Y-%m-%d")
        feat = pd.read_parquet(feature_store_directory + "gold_feature_store_" + a.replace('-', '_') + ".parquet")
        lab = pd.read_parquet(label_store_directory + "gold_label_store_" + label_date.replace('-', '_') + ".parquet")[["loan_id", "label"]]
        frames.append(feat.merge(lab, on="loan_id", how="inner"))
    df = pd.concat(frames, ignore_index=True)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def time_split(df):
    # latest months of the window are validation (at least 2 months, about a quarter of the window)
    months = sorted(df["snapshot_date"].unique())
    n_val = max(2, round(len(months) / 4))
    is_val = df["snapshot_date"].isin(months[-n_val:])
    return df[~is_val], df[is_val], n_val


def get_feature_columns(df):
    feats = [c for c in df.columns if c not in ID_COLS + ["label"]]
    cat_cols = [c for c in feats if df[c].dtype == "object"]
    num_cols = [c for c in feats if c not in cat_cols]
    return num_cols, cat_cols


def build_preprocessor(model_type, num_cols, cat_cols):
    # every step is fitted on the training rows only (the Pipeline is fitted inside fit_pipeline)
    cat_pipe = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    if model_type == "logreg":
        # quantile transform copes with the skewed money columns and outliers
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median")),
                             ("quantile", QuantileTransformer(n_quantiles=200, output_distribution="normal"))])
    else:
        num_pipe = "passthrough"  # XGBoost handles missing values and skew natively
    return ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])


def make_estimator(model_type, params, y_train):
    if model_type == "logreg":
        return LogisticRegression(C=params["C"], class_weight=params["class_weight"], max_iter=2000)
    spw = params["scale_pos_weight"]
    if spw == "balanced":
        spw = float((y_train == 0).sum() / (y_train == 1).sum())
    return XGBClassifier(max_depth=params["max_depth"], n_estimators=params["n_estimators"], scale_pos_weight=spw,
                         learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                         tree_method="hist", eval_metric="logloss", n_jobs=2, random_state=42)


def fit_pipeline(model_type, params, num_cols, cat_cols, X, y):
    pipe = Pipeline([("prep", build_preprocessor(model_type, num_cols, cat_cols)),
                     ("model", make_estimator(model_type, params, y))])
    pipe.fit(X, y)
    return pipe


def candidate_models():
    cands = []
    for c in [0.01, 0.1, 1.0]:
        for cw in [None, "balanced"]:
            cands.append(("logreg", {"C": c, "class_weight": cw}))
    for depth in [2, 3, 4]:
        for n_est in [150, 300]:
            for spw in [1.0, "balanced"]:
                cands.append(("xgboost", {"max_depth": depth, "n_estimators": n_est, "scale_pos_weight": spw}))
    return cands


def ks_stat(y_true, score):
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def pick_threshold(y_true, score, method="ks"):
    # loan is flagged high risk when score >= threshold
    if method == "ks":
        fpr, tpr, thr = roc_curve(y_true, score)
        return float(thr[np.argmax(tpr - fpr)])
    prec, rec, thr = precision_recall_curve(y_true, score)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[np.argmax(f1)])


def threshold_report(y_true, score, threshold):
    flag = score >= threshold
    return {"flagged_rate": round(float(flag.mean()), 3),
            "precision": round(float(y_true[flag].mean()), 3) if flag.any() else None,
            "recall": round(float(flag[y_true == 1].mean()), 3)}


def run_training(run_date_str, feature_store_directory, label_store_directory, model_store_directory, threshold_method="ks"):
    # 1. window and data
    window = get_training_window(run_date_str)
    df = load_training_data(window, feature_store_directory, label_store_directory)
    df_train, df_val, n_val = time_split(df)
    num_cols, cat_cols = get_feature_columns(df)
    print("run", run_date_str, "| window", window[0], "to", window[-1], f"({len(window)} months)",
          "| rows", len(df), "| default rate", round(df["label"].mean(), 3))
    print("train rows", len(df_train), "| validation rows", len(df_val), f"(last {n_val} months)")

    # 2. try every candidate, fit on train only, score on validation
    results = []
    for model_type, params in candidate_models():
        pipe = fit_pipeline(model_type, params, num_cols, cat_cols, df_train[num_cols + cat_cols], df_train["label"])
        score = pipe.predict_proba(df_val[num_cols + cat_cols])[:, 1]
        results.append({"model_type": model_type, "params": params,
                        "val_auc": roc_auc_score(df_val["label"], score),
                        "val_ks": ks_stat(df_val["label"], score), "score": score})
    results.sort(key=lambda r: r["val_auc"], reverse=True)
    best = results[0]
    print("top candidates (validation AUC):")
    for r in results[:5]:
        print("  ", r["model_type"], r["params"], "AUC", round(r["val_auc"], 4), "KS", round(r["val_ks"], 4))

    # 3. threshold chosen on validation scores of the best candidate
    y_val = df_val["label"].to_numpy()
    threshold = pick_threshold(y_val, best["score"], threshold_method)
    report = threshold_report(y_val, best["score"], threshold)
    print("best:", best["model_type"], best["params"], "| threshold", round(threshold, 4), threshold_method, report)

    # 4. refit the chosen recipe on the whole window, then save
    pipe = fit_pipeline(best["model_type"], best["params"], num_cols, cat_cols, df[num_cols + cat_cols], df["label"])
    version = "model_" + run_date_str.replace('-', '_')
    out_dir = model_store_directory + version + "/"
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pipe, out_dir + "model.pkl")
    metadata = {
        "model_version": version, "run_date": run_date_str, "trained_at": datetime.now(timezone.utc).isoformat(),
        "window_start": window[0], "window_end": window[-1], "window_months": len(window),
        "n_rows": int(len(df)), "default_rate": round(float(df["label"].mean()), 4),
        "validation_months": int(n_val), "model_type": best["model_type"], "params": best["params"],
        "val_auc": round(best["val_auc"], 4), "val_ks": round(best["val_ks"], 4),
        "threshold": threshold, "threshold_method": threshold_method, "threshold_report": report,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "leaderboard": [{"model_type": r["model_type"], "params": r["params"],
                         "val_auc": round(r["val_auc"], 4), "val_ks": round(r["val_ks"], 4)} for r in results[:10]],
    }
    with open(out_dir + "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("saved to:", out_dir)
    return metadata