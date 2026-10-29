import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pprint
import pyspark
import pyspark.sql.functions as F
import argparse

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_labels_gold_table(snapshot_date_str, silver_loan_daily_directory, gold_label_store_directory, spark, dpd, mob):
    
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to silver table
    partition_name = "silver_loan_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_loan_daily_directory + partition_name
    df = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df.count())

    # get customer at mob
    df = df.filter(col("mob") == mob)

    # get label
    df = df.withColumn("label", F.when(col("dpd") >= dpd, 1).otherwise(0).cast(IntegerType()))
    df = df.withColumn("label_def", F.lit(str(dpd)+'dpd_'+str(mob)+'mob').cast(StringType()))

    # select columns to save
    df = df.select("loan_id", "Customer_ID", "label", "label_def", "snapshot_date")

    # save gold table - IRL connect to database to write
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = gold_label_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df

def process_features_gold_table(snapshot_date_str, gold_label_store_directory, silver_clickstream_directory, \
                                                               silver_attributes_directory, silver_financials_directory, \
                                                               gold_feature_store_directory, spark, mob = 6):

    # 1. Read month label partition
    partition_name = "gold_label_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    df_label = spark.read.parquet(gold_label_store_directory + partition_name)

    # 2. Skip months where no loan has reached mob=6 yet
    if df_label.count() == 0:
        print(snapshot_date_str, 'skipped - no loans at mob =', mob, 'yet')
        return None

    # 3. Derive the application date from mob
    loan_start_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d") - relativedelta(months=mob)
    loan_start_date_str = loan_start_date.strftime("%Y-%m-%d")
    suffix = loan_start_date_str.replace('-','_') + '.parquet'

    # 4. Read the three silver sources at the loan application date, not snapshot_date_str
    df_click = spark.read.parquet(silver_clickstream_directory + "silver_clickstream_daily_" + suffix).drop("snapshot_date")
    df_attr  = spark.read.parquet(silver_attributes_directory + "silver_attributes_daily_" + suffix).drop("snapshot_date")
    df_fin   = spark.read.parquet(silver_financials_directory + "silver_financials_daily_" + suffix).drop("snapshot_date")

    # 5. Join, label as the anchor (left joins preserve every labeled loan)
    df = df_label.join(df_attr, on="Customer_ID", how="left")
    df = df.join(df_fin, on="Customer_ID", how="left")
    df = df.join(df_click, on="Customer_ID", how="left")

    # 6. Feature engineering #1 - has_clickstream flag
    #df = df.withColumn("has_clickstream", F.when(col("fe_1").isNotNull(), 1).otherwise(0))
    # decided to cancel this feature that apparently all records come with "1", as clickstream data only available from 2023-01-01
    # to 2024-06-01 (first 18 months), while the last 7 months aren't there, which anyway being excluded due to MOB

    # 7. Feature engineering #2 - based on Singapore's MOM person lifecycle 
    df = df.withColumn("age_bucket",
    F.when(col("Age").isNull(), "Unknown")
     .when((col("Age") >= 15) & (col("Age") <= 24), "15-24_Pre_Work_Early_Entrants")
     .when((col("Age") >= 25) & (col("Age") <= 39), "25-39_Core_Active_Workforce")
     .when((col("Age") >= 40) & (col("Age") <= 54), "40-54_Mature_Workforce")
     .when((col("Age") >= 55) & (col("Age") <= 63), "55-63_Sr_Active_Workforce")
     .when((col("Age") >= 64) & (col("Age") <= 69), "64-69_Statutory_Reemp_Window")
     .when(col("Age") >= 70, "70_Post_Reemp_Silver_Workforce")
     .otherwise("Unknown"))
    df = df.drop("Age") # drop as it's now represented by the Age Bucket

    # 8. Drop label columns as label available in different datamart
    df = df.drop("label", "label_def")

    # 9. Save
    partition_name = "gold_feature_store_" + snapshot_date_str.replace('-','_') + '.parquet'
    df.write.mode("overwrite").parquet(gold_feature_store_directory + partition_name)
    print('saved to:', gold_feature_store_directory + partition_name)
    
    return df