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


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_financials_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data first so we can put the right data type
    
    ## Group-1 string trailing "_"
    df = df.withColumn("Annual_Income", F.regexp_replace(col("Annual_Income"), "_$", "").cast(FloatType()))
    # df = df.withColumn("Num_of_Loan", F.regexp_replace(col("Num_of_Loan"), "_$", "").cast(IntegerType()))
    df = df.withColumn("Num_of_Delayed_Payment", F.regexp_replace(col("Num_of_Delayed_Payment"), "_$", "").cast(IntegerType()))
    df = df.withColumn("Outstanding_Debt", F.regexp_replace(col("Outstanding_Debt"), "_$", "").cast(FloatType()))

    ## Group-2 capping to reasonable value
    df = df.withColumn("Num_Bank_Accounts", F.when((col("Num_Bank_Accounts") < 0) | (col("Num_Bank_Accounts") > 20), None).otherwise(col("Num_Bank_Accounts")))
    df = df.withColumn("Num_Credit_Card", F.when(col("Num_Credit_Card") > 20, None).otherwise(col("Num_Credit_Card")))
    df = df.withColumn("Interest_Rate", F.when(col("Interest_Rate") > 50, None).otherwise(col("Interest_Rate")))
    # df = df.withColumn("Num_of_Loan", F.when((col("Num_of_Loan") < 0) | (col("Num_of_Loan") > 10), None).otherwise(col("Num_of_Loan")))
    df = df.withColumn("Num_of_Loan",
    F.when(col("Type_of_Loan").isNull(), 0)
     .otherwise(F.size(F.split(col("Type_of_Loan"), ",")))
                      )
    df = df.drop("Type_of_Loan") # drop as now it's used only to derive Num_of_Loan
    df = df.withColumn("Delay_from_due_date", F.when(col("Delay_from_due_date") < 0, None).otherwise(col("Delay_from_due_date")))
    df = df.withColumn("Num_of_Delayed_Payment", F.when((col("Num_of_Delayed_Payment") < 0) | (col("Num_of_Delayed_Payment") > 36),\
                                                        None).otherwise(col("Num_of_Delayed_Payment")))
    df = df.withColumn("Num_Credit_Inquiries", F.when(col("Num_Credit_Inquiries") > 50, None).otherwise(col("Num_Credit_Inquiries")))

    ## Group-3 change "_" to null/none
    df = df.withColumn("Changed_Credit_Limit", F.when(col("Changed_Credit_Limit") == "_", None).otherwise(col("Changed_Credit_Limit")).cast(FloatType()))
    df = df.withColumn("Credit_Mix", F.when(col("Credit_Mix") == "_", None).otherwise(col("Credit_Mix")))

    ## Group-4 unclear string/characters change to null
    df = df.withColumn("Payment_Behaviour", F.when(col("Payment_Behaviour") == "!@9#%8", None).otherwise(col("Payment_Behaviour")))

    ## Group-5 A pattern of "__X__" to remove the "__" character
    df = df.withColumn("Amount_invested_monthly", F.when(col("Amount_invested_monthly").rlike("^__.*__$"),None) \
                       .otherwise(col("Amount_invested_monthly")).cast(FloatType()))
    df = df.withColumn("Monthly_Balance", F.when(col("Monthly_Balance").rlike("^__.*__$"), None).otherwise(col("Monthly_Balance")).cast(FloatType()))
    
    ## Group-6 Specifically of Credit_History_Age to extract the value from string to numbers
    df = df.withColumn("Credit_History_Age_Months",
    F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Years", 1).cast(IntegerType()) * 12 +
    F.regexp_extract(col("Credit_History_Age"), r"(\d+)\s*Months", 1).cast(IntegerType())).drop("Credit_History_Age")

    ## Group-7 Total_EMI_per_month to sound logical
    df = df.withColumn("Total_EMI_per_month", F.when(col("Total_EMI_per_month") > col("Monthly_Inhand_Salary"), None).otherwise(col("Total_EMI_per_month")))
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df