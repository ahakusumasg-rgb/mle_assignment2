import argparse

import utils.model_promotion


def main(snapshotdate, model_store, min_val_auc):
    print("\n\n---starting job---\n\n")
    utils.model_promotion.run_promotion(snapshotdate, model_store, min_val_auc)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="promote the candidate trained for the run date if it passes the sanity checks")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date D (YYYY-MM-DD)")
    parser.add_argument("--model-store", type=str, default="model_store/")
    parser.add_argument("--min-val-auc", type=float, default=0.70, help="validation AUC floor for auto-promotion")
    args = parser.parse_args()
    main(args.snapshotdate, args.model_store, args.min_val_auc)