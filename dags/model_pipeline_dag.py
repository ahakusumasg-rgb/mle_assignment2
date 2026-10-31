"""Monthly loan default pipeline (CS611 Assignment 2).

bronze_to_silver -> gold_labels / gold_features -> score (2024-01..06) -> monitor -> check_trigger -> train -> promote.
Date gates (ShortCircuit) decide which months score, monitor and train, refer to the governance SOP.
"""
import json
import os
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.empty import EmptyOperator
from airflow.providers.standard.operators.python import BranchPythonOperator, PythonOperator, ShortCircuitOperator

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", "/opt/airflow")

# date gates (run months, first of the month)
FEATURES_END = "2024-06-01"                      # clickstream and new applications stop after 2024-06
SCORE_START, SCORE_END = "2024-01-01", "2024-06-01"   # score applications with the model trained in an earlier run
MONITOR_START = "2024-01-01"                     # drift from 2024-01, performance from 2024-07 (labels arrive 6 months later)
TRAIN_START, TRAIN_END = "2023-12-01", "2024-06-01"   # v1 needs 6 matured months; the 12 month window is reached in 2024-06


def script(name, *extra):
    # every task calls a script from the project root with the run month
    return f"cd {PROJECT_ROOT} && python src/{name} --snapshotdate {{{{ ds }}}} " + " ".join(extra)


def in_window(start, end, ds=None, **_):
    return start <= ds <= end


def decide_branch(ds=None, **_):
    import utils.model_monitoring
    fired = utils.model_monitoring.read_trigger(ds, "datamart/gold/monitoring/")
    return "adhoc_review" if fired else "no_action"


def record_adhoc_trigger(ds=None, **_):
    # phase 1: record that the champion tripped a trigger; a human decides what to do (see the governance SOP)
    import pandas as pd
    import utils.model_monitoring as mm
    path = mm.partition_path("datamart/gold/monitoring/", "gold_monitoring_summary_", ds)
    row = pd.read_parquet(path).query("arm == 'champion'").iloc[0]
    clean = lambda v: None if pd.isna(v) else float(v)    # NaN is not valid JSON
    record = {"run_month": ds, "status": row["status"], "psi_alert": bool(row["psi_alert"]),
              "perf_trigger": bool(row["perf_trigger"]), "score_psi": clean(row["score_psi"]),
              "auc_rolling3": clean(row["auc_rolling3"]), "auc_alert_level": clean(row["auc_alert_level"])}
    os.makedirs("model_store", exist_ok=True)
    with open("model_store/adhoc_trigger_log.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print("AD-HOC REVIEW REQUIRED:", record)


with DAG(
    dag_id="model_pipeline",
    description="Monthly loan default pipeline: data, score, monitor, train, promote",
    schedule="0 0 1 * *",
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 1),
    catchup=True,
    max_active_runs=1,
    default_args={"retries": 1},
    tags=["cs611", "loan-default"],
    doc_md=__doc__,
) as dag:

    bronze_to_silver = BashOperator(task_id="bronze_to_silver", bash_command=script("data_pipeline.py", "--stage bronze_silver"))
    gold_labels = BashOperator(task_id="gold_labels", bash_command=script("data_pipeline.py", "--stage gold_labels"))

    gate_features = ShortCircuitOperator(task_id="gate_features", python_callable=in_window,
                                         op_kwargs={"start": "0000-00-00", "end": FEATURES_END}, ignore_downstream_trigger_rules=False)
    gold_features = BashOperator(task_id="gold_features", bash_command=script("data_pipeline.py", "--stage gold_features"))

    gate_score = ShortCircuitOperator(task_id="gate_score", python_callable=in_window,
                                      op_kwargs={"start": SCORE_START, "end": SCORE_END}, ignore_downstream_trigger_rules=False)
    score = BashOperator(task_id="score", bash_command=script("score_model.py"))

    gate_monitor = ShortCircuitOperator(task_id="gate_monitor", python_callable=in_window, trigger_rule="none_failed",
                                        op_kwargs={"start": MONITOR_START, "end": "9999-99-99"}, ignore_downstream_trigger_rules=False)
    monitor = BashOperator(task_id="monitor", bash_command=script("monitor_model.py"))
    monitor_report = BashOperator(task_id="monitor_report", bash_command=f"cd {PROJECT_ROOT} && python src/monitor_report.py")

    check_trigger = BranchPythonOperator(task_id="check_trigger", python_callable=decide_branch)
    adhoc_review = PythonOperator(task_id="adhoc_review", python_callable=record_adhoc_trigger)
    no_action = EmptyOperator(task_id="no_action")
    trigger_done = EmptyOperator(task_id="trigger_done", trigger_rule="none_failed_min_one_success")

    gate_train = ShortCircuitOperator(task_id="gate_train", python_callable=in_window, trigger_rule="none_failed",
                                      op_kwargs={"start": TRAIN_START, "end": TRAIN_END}, ignore_downstream_trigger_rules=False)
    train = BashOperator(task_id="train", bash_command=script("train_model.py"))
    promote = BashOperator(task_id="promote", bash_command=script("promote_model.py"))

    bronze_to_silver >> [gold_labels, gate_features]
    gate_features >> gold_features >> gate_score >> score
    [score, gold_labels] >> gate_monitor >> monitor >> [monitor_report, check_trigger]
    check_trigger >> [adhoc_review, no_action] >> trigger_done
    [monitor_report, trigger_done] >> gate_train >> train >> promote