import argparse

import utils.model_monitoring


def main(snapshotdate, feature_store, label_store, predictions_dir, model_store, monitoring_dir, auc_tolerance):
    print("\n\n---starting job---\n\n")
    utils.model_monitoring.run_monitoring(snapshotdate, feature_store, label_store, predictions_dir,
                                          model_store, monitoring_dir, auc_tolerance)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor drift (PSI) and performance (AUC, KS) for the run month and set trigger flags")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date D (YYYY-MM-DD)")
    parser.add_argument("--feature-store", type=str, default="datamart/gold/feature_store/")
    parser.add_argument("--label-store", type=str, default="datamart/gold/label_store/")
    parser.add_argument("--predictions-dir", type=str, default="datamart/gold/predictions/")
    parser.add_argument("--model-store", type=str, default="model_store/")
    parser.add_argument("--monitoring-dir", type=str, default="datamart/gold/monitoring/")
    parser.add_argument("--auc-tolerance", type=float, default=0.05, help="allowed drop of live rolling AUC below validation AUC")
    args = parser.parse_args()
    main(args.snapshotdate, args.feature_store, args.label_store, args.predictions_dir,
         args.model_store, args.monitoring_dir, args.auc_tolerance)