import os
import numpy as np
import pandas as pd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score

import utils.model_promotion
import utils.model_scoring
import utils.model_training

MOB = 6                  # outcome of a loan is known 6 months after application
PSI_WATCH = 0.10         # fixed before looking at results (industry convention)
PSI_ALERT = 0.25
AUC_TOLERANCE = 0.05     # live rolling AUC may sit this far below the model's validation AUC
ROLLING_COHORTS = 3      # cohorts pooled for the rolling AUC
N_BINS = 10
EPS = 1e-4


def shift_month(date_str, months):
    return (datetime.strptime(date_str, "%Y-%m-%d") + relativedelta(months=months)).strftime("%Y-%m-%d")


def partition_path(directory, prefix, date_str):
    return directory + prefix + date_str.replace('-', '_') + ".parquet"


# ---------- PSI ----------

def _psi(expected, actual):
    e = np.clip(expected, EPS, None)
    a = np.clip(actual, EPS, None)
    return float(((a - e) * np.log(a / e)).sum())


def psi_categorical(ref, cur):
    # one bin per category, missing values form their own bin, unseen categories get the floor value
    r = pd.Series(ref).fillna("Missing").astype(str).value_counts(normalize=True)
    c = pd.Series(cur).fillna("Missing").astype(str).value_counts(normalize=True)
    idx = r.index.union(c.index)
    return _psi(r.reindex(idx).fillna(0).to_numpy(), c.reindex(idx).fillna(0).to_numpy())


def _bin_proportions(values, inner_edges):
    values = pd.Series(values)
    counts = np.bincount(np.digitize(values.dropna(), inner_edges), minlength=len(inner_edges) + 1)
    return np.append(counts, values.isna().sum()) / max(len(values), 1)   # last bin = missing


def psi_numeric(ref, cur, n_bins=N_BINS):
    # bins are the deciles of the reference, so each reference bin holds about 10% by construction
    ref = pd.Series(ref)
    if ref.nunique() <= n_bins:
        return psi_categorical(ref, cur)   # few distinct values: treat as categories
    inner = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, n_bins + 1))[1:-1])
    return _psi(_bin_proportions(ref, inner), _bin_proportions(cur, inner))


def psi_status(psi):
    if psi >= PSI_ALERT:
        return "alert"
    if psi >= PSI_WATCH:
        return "watch"
    return "ok"


# ---------- drift (no labels needed, available in the run month) ----------

def load_reference(meta, model_store_directory, feature_store_directory):
    # reference = the model's own training window: its features and the scores the model gives them
    months = [d.strftime("%Y-%m-%d") for d in pd.date_range(meta["window_start"], meta["window_end"], freq="MS")]
    ref = pd.concat([pd.read_parquet(partition_path(feature_store_directory, "gold_feature_store_", m)) for m in months],
                    ignore_index=True)
    score, _ = utils.model_scoring.score_with_model(ref, model_store_directory, meta["model_version"])
    return ref, score


def compute_drift(run_date_str, preds, feature_store_directory, model_store_directory):
    cur_features = pd.read_parquet(partition_path(feature_store_directory, "gold_feature_store_", run_date_str))
    refs, rows = {}, []
    for arm in ["champion", "control"]:
        g = preds[preds["arm"] == arm]
        version = g["model_version"].iloc[0]
        meta = utils.model_promotion.load_metadata(model_store_directory, version)
        if version not in refs:
            refs[version] = load_reference(meta, model_store_directory, feature_store_directory)
        ref_df, ref_score = refs[version]
        base = {"run_month": run_date_str, "cohort": run_date_str, "arm": arm, "model_version": version}
        rows.append({**base, "item_type": "score", "item": "score",
                     "psi": psi_numeric(ref_score, g["score"].astype(float))})
        if arm == "champion":   # feature drift is measured once, against the champion's training window
            for c in meta["num_cols"]:
                rows.append({**base, "item_type": "feature", "item": c, "psi": psi_numeric(ref_df[c], cur_features[c])})
            for c in meta["cat_cols"]:
                rows.append({**base, "item_type": "feature", "item": c, "psi": psi_categorical(ref_df[c], cur_features[c])})
    df = pd.DataFrame(rows)
    df["status"] = df["psi"].map(psi_status)
    return df


# ---------- performance (needs labels, available 6 months after scoring) ----------

def load_cohort_with_labels(cohort_str, predictions_directory, label_store_directory):
    pred_path = partition_path(predictions_directory, "gold_predictions_", cohort_str)
    label_path = partition_path(label_store_directory, "gold_label_store_", shift_month(cohort_str, MOB))
    if not (os.path.exists(pred_path) and os.path.exists(label_path)):
        return None
    labels = pd.read_parquet(label_path)[["loan_id", "label"]]
    return pd.read_parquet(pred_path).merge(labels, on="loan_id", how="inner")


def safe_auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def performance_metrics(g):
    y = g["label"].to_numpy()
    s = g["score"].to_numpy().astype(float)
    flag = g["prediction"].to_numpy() == 1
    return {"n_loans": int(len(g)), "default_rate": float(y.mean()), "auc": safe_auc(y, s),
            "ks": utils.model_training.ks_stat(y, s) if len(np.unique(y)) == 2 else np.nan,
            "flagged_rate": float(flag.mean()),
            "precision": float(y[flag].mean()) if flag.any() else np.nan,
            "recall": float(flag[y == 1].mean()) if (y == 1).any() else np.nan}


def compute_performance(run_date_str, predictions_directory, label_store_directory, model_store_directory, auc_tolerance):
    cohort = shift_month(run_date_str, -MOB)       # the cohort whose outcomes just arrived
    current = load_cohort_with_labels(cohort, predictions_directory, label_store_directory)
    if current is None:
        return cohort, {}
    window = [load_cohort_with_labels(shift_month(cohort, -k), predictions_directory, label_store_directory)
              for k in reversed(range(ROLLING_COHORTS))]
    val_auc = lambda v: utils.model_promotion.load_metadata(model_store_directory, v)["val_auc"]
    out = {}
    for arm, g in current.groupby("arm"):
        version = g["model_version"].iloc[0]
        m = {"perf_cohort": cohort, "perf_model_version": version, "val_auc": val_auc(version), **performance_metrics(g)}
        if all(w is not None for w in window):
            parts = [w[w["arm"] == arm] for w in window]
            pooled = pd.concat(parts)
            expected = sum(len(p) * val_auc(p["model_version"].iloc[0]) for p in parts) / len(pooled)
            m["auc_rolling3"] = safe_auc(pooled["label"], pooled["score"].astype(float))
            m["expected_auc_rolling3"] = float(expected)
            m["auc_alert_level"] = float(expected - auc_tolerance)
            m["perf_breach"] = bool(m["auc_rolling3"] < m["auc_alert_level"])
        out[arm] = m
    return cohort, out


def previous_breach(run_date_str, monitoring_directory):
    path = partition_path(monitoring_directory, "gold_monitoring_summary_", shift_month(run_date_str, -1))
    if not os.path.exists(path):
        return {}
    prev = pd.read_parquet(path)
    return dict(zip(prev["arm"], prev["perf_breach"].fillna(False).astype(bool)))


# ---------- orchestration ----------

SUMMARY_COLS = ["run_month", "arm", "drift_cohort", "drift_model_version", "score_psi", "score_psi_status",
                "max_feature_psi", "max_feature", "n_feature_watch", "n_feature_alert",
                "perf_cohort", "perf_model_version", "val_auc", "n_loans", "default_rate", "auc", "ks",
                "flagged_rate", "precision", "recall", "auc_rolling3", "expected_auc_rolling3", "auc_alert_level",
                "perf_breach", "perf_trigger", "psi_alert", "retrain_trigger", "status"]


def run_monitoring(run_date_str, feature_store_directory, label_store_directory, predictions_directory,
                   model_store_directory, monitoring_directory, auc_tolerance=AUC_TOLERANCE):
    # 1. drift on the cohort scored this month (application month = run month)
    drift = None
    pred_path = partition_path(predictions_directory, "gold_predictions_", run_date_str)
    if os.path.exists(pred_path):
        drift = compute_drift(run_date_str, pd.read_parquet(pred_path), feature_store_directory, model_store_directory)

    # 2. performance on the cohort whose outcomes just arrived (application month = run month - 6)
    perf_cohort, perf = compute_performance(run_date_str, predictions_directory, label_store_directory,
                                            model_store_directory, auc_tolerance)

    if drift is None and not perf:
        print(run_date_str, "skipped - nothing to monitor (no cohort scored this month, no outcomes landed)")
        return None

    # 3. one summary row per arm, with trigger flags
    prev = previous_breach(run_date_str, monitoring_directory)
    rows = []
    for arm in ["champion", "control"]:
        row = {c: np.nan for c in SUMMARY_COLS}
        row.update({"run_month": run_date_str, "arm": arm})
        if drift is not None:
            d = drift[drift["arm"] == arm]
            s = d[d["item_type"] == "score"].iloc[0]
            row.update({"drift_cohort": s["cohort"], "drift_model_version": s["model_version"],
                        "score_psi": s["psi"], "score_psi_status": s["status"]})
            f = drift[drift["item_type"] == "feature"]
            if arm == "champion" and len(f):
                top = f.loc[f["psi"].idxmax()]
                row.update({"max_feature_psi": top["psi"], "max_feature": top["item"],
                            "n_feature_watch": int((f["status"] == "watch").sum()),
                            "n_feature_alert": int((f["status"] == "alert").sum())})
        row.update(perf.get(arm, {}))
        breach = bool(row["perf_breach"]) if not pd.isna(row["perf_breach"]) else False
        row["perf_breach"] = breach
        row["perf_trigger"] = breach and prev.get(arm, False)      # breach this month and last month
        row["psi_alert"] = bool(row["score_psi_status"] == "alert" or (row["n_feature_alert"] > 0))
        row["retrain_trigger"] = bool(row["perf_trigger"] or row["psi_alert"])
        watch = row["score_psi_status"] == "watch" or row["n_feature_watch"] > 0
        row["status"] = "alert" if row["retrain_trigger"] else ("watch" if (watch or breach) else "ok")
        rows.append(row)
    summary = pd.DataFrame(rows)[SUMMARY_COLS]

    # 4. one partition per run month; rerunning a month overwrites only that month
    os.makedirs(monitoring_directory, exist_ok=True)
    summary.to_parquet(partition_path(monitoring_directory, "gold_monitoring_summary_", run_date_str), index=False)
    if drift is not None:
        drift.to_parquet(partition_path(monitoring_directory, "gold_monitoring_drift_", run_date_str), index=False)

    for _, r in summary.iterrows():
        bits = []
        if not pd.isna(r["score_psi"]):
            bits.append(f"cohort {r['drift_cohort'][:7]} score PSI {r['score_psi']:.3f} ({r['score_psi_status']})")
            if not pd.isna(r["max_feature_psi"]):
                bits.append(f"max feature PSI {r['max_feature_psi']:.3f} ({r['max_feature']})")
        if not pd.isna(r["auc"]):
            bits.append(f"cohort {r['perf_cohort'][:7]} AUC {r['auc']:.3f} KS {r['ks']:.3f} n={int(r['n_loans'])}")
        if not pd.isna(r["auc_rolling3"]):
            bits.append(f"rolling3 AUC {r['auc_rolling3']:.3f} vs alert level {r['auc_alert_level']:.3f}")
        print(f"{r['arm']:<8} {r['status'].upper():<5} | " + " | ".join(bits) + f" | retrain_trigger={r['retrain_trigger']}")
    print("saved to:", monitoring_directory)
    return summary


def read_trigger(run_date_str, monitoring_directory):
    # used by the DAG branch: did the champion trip a retrain trigger this run month?
    path = partition_path(monitoring_directory, "gold_monitoring_summary_", run_date_str)
    if not os.path.exists(path):
        return False
    df = pd.read_parquet(path)
    return bool(df.loc[df["arm"] == "champion", "retrain_trigger"].any())