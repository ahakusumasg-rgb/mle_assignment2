import argparse

import utils.model_scoring


def main(snapshotdate, feature_store, model_store, predictions_dir):
    print("\n\n---starting job---\n\n")
    utils.model_scoring.run_scoring(snapshotdate, feature_store, model_store, predictions_dir)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score the loans applied for in the run month with the champion and the control model")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date D (YYYY-MM-DD), also the application month scored")
    parser.add_argument("--feature-store", type=str, default="datamart/gold/feature_store/")
    parser.add_argument("--model-store", type=str, default="model_store/")
    parser.add_argument("--predictions-dir", type=str, default="datamart/gold/predictions/")
    args = parser.parse_args()
    main(args.snapshotdate, args.feature_store, args.model_store, args.predictions_dir)