import os
import argparse
import pyspark

import utils.data_processing_bronze_table
import utils.data_processing_silver_table
import utils.data_processing_silver_clickstream_table
import utils.data_processing_silver_attributes_table
import utils.data_processing_silver_financials_table
import utils.data_processing_gold_table

# same layout as main.py, paths are relative to the project root
BRONZE_LMS = "datamart/bronze/lms/"
BRONZE_CLICKSTREAM = "datamart/bronze/clickstream/"
BRONZE_ATTRIBUTES = "datamart/bronze/attributes/"
BRONZE_FINANCIALS = "datamart/bronze/financials/"
SILVER_LMS = "datamart/silver/lms/"
SILVER_CLICKSTREAM = "datamart/silver/clickstream/"
SILVER_ATTRIBUTES = "datamart/silver/attributes/"
SILVER_FINANCIALS = "datamart/silver/financials/"
GOLD_LABEL_STORE = "datamart/gold/label_store/"
GOLD_FEATURE_STORE = "datamart/gold/feature_store/"


def get_spark():
    spark = pyspark.sql.SparkSession.builder \
        .appName("data_pipeline") \
        .master("local[*]") \
        .config("spark.ui.enabled", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def make_dirs(directories):
    for d in directories:
        os.makedirs(d, exist_ok=True)


def run_bronze_silver(snapshotdate, spark):
    bronze_directories = [BRONZE_LMS, BRONZE_CLICKSTREAM, BRONZE_ATTRIBUTES, BRONZE_FINANCIALS]
    make_dirs(bronze_directories + [SILVER_LMS, SILVER_CLICKSTREAM, SILVER_ATTRIBUTES, SILVER_FINANCIALS])
    for d in bronze_directories:
        utils.data_processing_bronze_table.process_bronze_table(snapshotdate, d, spark)
    utils.data_processing_silver_table.process_silver_table(snapshotdate, BRONZE_LMS, SILVER_LMS, spark)
    utils.data_processing_silver_clickstream_table.process_silver_clickstream_table(snapshotdate, BRONZE_CLICKSTREAM, SILVER_CLICKSTREAM, spark)
    utils.data_processing_silver_attributes_table.process_silver_attributes_table(snapshotdate, BRONZE_ATTRIBUTES, SILVER_ATTRIBUTES, spark)
    utils.data_processing_silver_financials_table.process_silver_financials_table(snapshotdate, BRONZE_FINANCIALS, SILVER_FINANCIALS, spark)


def run_gold_labels(snapshotdate, spark):
    make_dirs([GOLD_LABEL_STORE])
    utils.data_processing_gold_table.process_labels_gold_table(snapshotdate, SILVER_LMS, GOLD_LABEL_STORE, spark, dpd=30, mob=6)


def run_gold_features(snapshotdate, spark):
    make_dirs([GOLD_FEATURE_STORE])
    utils.data_processing_gold_table.process_features_gold_table(snapshotdate, SILVER_LMS, SILVER_CLICKSTREAM,
                                                                 SILVER_ATTRIBUTES, SILVER_FINANCIALS, GOLD_FEATURE_STORE, spark)


STAGES = {"bronze_silver": run_bronze_silver, "gold_labels": run_gold_labels, "gold_features": run_gold_features}


def main(snapshotdate, stage):
    print("\n\n---starting job---\n\n")
    spark = get_spark()
    STAGES[stage](snapshotdate, spark)
    spark.stop()
    print("\n\n---completed job---\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="data stage for one run month: bronze + silver, gold labels or gold features")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date D (YYYY-MM-DD)")
    parser.add_argument("--stage", type=str, required=True, choices=list(STAGES))
    args = parser.parse_args()
    main(args.snapshotdate, args.stage)