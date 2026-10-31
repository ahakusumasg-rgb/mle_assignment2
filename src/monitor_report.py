import argparse

import utils.monitoring_report


def main(monitoring_dir, output):
    print("\n\n---starting job---\n\n")
    utils.monitoring_report.build_report(monitoring_dir, output)
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build the HTML monitoring report from the gold monitoring tables")
    parser.add_argument("--monitoring-dir", type=str, default="datamart/gold/monitoring/")
    parser.add_argument("--output", type=str, default="datamart/gold/monitoring/monitoring_report.html")
    args = parser.parse_args()
    main(args.monitoring_dir, args.output)