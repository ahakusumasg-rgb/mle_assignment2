import os
import json
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

MOB = 6
MIN_VAL_AUC = 0.70
MIN_WINDOW_MONTHS = 6
FLAGGED_RATE_RANGE = (0.05, 0.80)
CHAMPION_FILE = "champion.json"
HISTORY_FILE = "promotion_history.jsonl"


def read_champion(model_store_directory):
    # champion.json is the pointer: which version scores new loans, and which version is the frozen control
    path = model_store_directory + CHAMPION_FILE
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_metadata(model_store_directory, version):
    with open(model_store_directory + version + "/metadata.json") as f:
        return json.load(f)


def sanity_checks(meta, run_date_str, model_store_directory, min_val_auc=MIN_VAL_AUC):
    # returns the list of failed checks (empty list = candidate may be promoted)
    failures = []
    expected_end = (datetime.strptime(run_date_str, "%Y-%m-%d") - relativedelta(months=MOB)).strftime("%Y-%m-%d")
    if not os.path.exists(model_store_directory + meta["model_version"] + "/model.pkl"):
        failures.append("model.pkl is missing")
    if meta["val_auc"] < min_val_auc:
        failures.append(f"validation AUC {meta['val_auc']} is below the floor {min_val_auc}")
    if not 0 < meta["threshold"] < 1:
        failures.append(f"threshold {meta['threshold']:.3f} is outside (0, 1)")
    flagged = meta["threshold_report"]["flagged_rate"]
    if not FLAGGED_RATE_RANGE[0] <= flagged <= FLAGGED_RATE_RANGE[1]:
        failures.append(f"flagged rate {flagged} is outside {FLAGGED_RATE_RANGE}")
    if meta["window_end"] != expected_end:
        failures.append(f"training window ends {meta['window_end']}, expected {expected_end}")
    if meta["window_months"] < MIN_WINDOW_MONTHS:
        failures.append(f"window has {meta['window_months']} months, minimum is {MIN_WINDOW_MONTHS}")
    return failures


def append_history(model_store_directory, record):
    # append-only audit trail of every promotion decision
    with open(model_store_directory + HISTORY_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_promotion(run_date_str, model_store_directory, min_val_auc=MIN_VAL_AUC):
    candidate = "model_" + run_date_str.replace('-', '_')
    champion = read_champion(model_store_directory)
    meta = load_metadata(model_store_directory, candidate)
    now = datetime.now(timezone.utc).isoformat()

    # idempotent: re-running the same month changes nothing
    if champion is not None and champion["champion"] == candidate:
        print(candidate, "is already the champion, nothing to do")
        return {"decision": "unchanged", "champion": candidate}

    failures = sanity_checks(meta, run_date_str, model_store_directory, min_val_auc)
    record = {"run_date": run_date_str, "candidate": candidate, "decided_at": now,
              "val_auc": meta["val_auc"], "previous_champion": champion["champion"] if champion else None}

    if failures:
        record.update({"decision": "rejected", "reasons": failures})
        append_history(model_store_directory, record)
        print(candidate, "REJECTED:", "; ".join(failures))
        print("champion stays:", champion["champion"] if champion else "none")
        return {"decision": "rejected", "reasons": failures, "champion": champion["champion"] if champion else None}

    if champion is None:
        # bootstrap: first accepted model is both the champion and the frozen control arm
        pointer = {"champion": candidate, "control": candidate, "promoted_run_date": run_date_str, "updated_at": now}
        record.update({"decision": "bootstrap"})
    else:
        pointer = {"champion": candidate, "control": champion["control"], "promoted_run_date": run_date_str, "updated_at": now}
        record.update({"decision": "promoted"})

    with open(model_store_directory + CHAMPION_FILE, "w") as f:
        json.dump(pointer, f, indent=2)
    append_history(model_store_directory, record)
    print(candidate, record["decision"].upper(), "| champion:", pointer["champion"], "| control:", pointer["control"])
    return {"decision": record["decision"], "champion": candidate}