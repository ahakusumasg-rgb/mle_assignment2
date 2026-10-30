import argparse

import utils.model_training


def main(snapshotdate, feature_store, label_store, model_store, threshold_method):
    print("\n\n---starting job---\n\n")
    utils.model_training.run_training(snapshotdate, feature_store, label_store, model_store, threshold_method)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train a candidate model for the run date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date D (YYYY-MM-DD)")
    parser.add_argument("--feature-store", type=str, default="datamart/gold/feature_store/")
    parser.add_argument("--label-store", type=str, default="datamart/gold/label_store/")
    parser.add_argument("--model-store", type=str, default="model_store/")
    parser.add_argument("--threshold-method", type=str, default="ks", choices=["ks", "f1"])
    args = parser.parse_args()
    main(args.snapshotdate, args.feature_store, args.label_store, args.model_store, args.threshold_method)