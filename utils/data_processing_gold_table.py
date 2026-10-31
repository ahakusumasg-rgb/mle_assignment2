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

CLICK_HISTORY_MONTHS = 12   # clickstream is averaged over this many monthly snapshots, ending at the application month - adopt from past work given by Prof.

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

def process_features_gold_table(snapshot_date_str, silver_loan_daily_directory, silver_clickstream_directory, \
                                                   silver_attributes_directory, silver_financials_directory, \
                                                   gold_feature_store_directory, spark):

    # snapshot_date_str is the APPLICATION month (A). No label store needed, so this also works at scoring time
    suffix = snapshot_date_str.replace('-','_') + '.parquet'

    # 1. Anchor: loans that start this month (mob 0 = application month)
    df_loan = spark.read.parquet(silver_loan_daily_directory + "silver_loan_daily_" + suffix)
    df = df_loan.filter(col("mob") == 0).select("loan_id", "Customer_ID", "snapshot_date")

    # 2. Skip months with no new loans
    if df.count() == 0:
        print(snapshot_date_str, 'skipped - no loans start this month')
        return None

    # 3. Read the three silver sources at the same application month
    df_click = spark.read.parquet(silver_clickstream_directory + "silver_clickstream_daily_" + suffix).drop("snapshot_date")
    df_attr  = spark.read.parquet(silver_attributes_directory + "silver_attributes_daily_" + suffix).drop("snapshot_date")
    df_fin   = spark.read.parquet(silver_financials_directory + "silver_financials_daily_" + suffix).drop("snapshot_date")

    # 3b. Clickstream history: mean of each column over the last 12 monthly snapshots up to and including the
    # application month. Only months <= snapshot_date are read, so nothing from the future can leak in
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    history_paths = []
    for k in range(CLICK_HISTORY_MONTHS):
        month_str = (snapshot_date - relativedelta(months=k)).strftime("%Y_%m_%d")
        path = silver_clickstream_directory + "silver_clickstream_daily_" + month_str + ".parquet"
        if os.path.exists(path):
            history_paths.append(path)
    df_history = spark.read.parquet(*history_paths)
    fe_cols = [c for c in df_history.columns if c.startswith("fe_")]
    df_history = df_history.groupBy("Customer_ID").agg(
        *[F.avg(c).alias("fe_avg12_" + c.replace("fe_", "")) for c in fe_cols],
        F.count("*").alias("click_history_months"))

    # 4. Join, new loans are the anchor (left joins preserve every loan)
    df = df.join(df_attr, on="Customer_ID", how="left")
    df = df.join(df_fin, on="Customer_ID", how="left")
    df = df.join(df_click, on="Customer_ID", how="left")
    df = df.join(df_history, on="Customer_ID", how="left")

    # 5. Feature engineering #1 - based on Singapore's MOM person lifecycle
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

    # 6. Feature engineering #2 - affordability and income scale
    df = df.withColumn("emi_to_salary",F.when(col("Monthly_Inhand_Salary") > 0, col("Total_EMI_per_month") / col("Monthly_Inhand_Salary")))
    df = df.withColumn("log_annual_income", F.log1p(col("Annual_Income")))

    # 7. Save, one partition per application month
    partition_name = "gold_feature_store_" + suffix
    df.write.mode("overwrite").parquet(gold_feature_store_directory + partition_name)
    print('saved to:', gold_feature_store_directory + partition_name)

    return df