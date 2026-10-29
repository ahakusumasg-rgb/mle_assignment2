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


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to bronze table
    partition_name = "bronze_attributes_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=True)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data first so we can put the right data type
    df = df.drop("Name", "SSN") # drop name/ssn from table as it isn't useful in ML, PII concern, and unique information already covered in customer_id
    df = df.withColumn("Occupation", F.when(col("Occupation") == "_______", None).otherwise(col("Occupation"))) # change to unknown
                                                                                                                     # maybe people with unknown tend to default
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_$", "").cast(IntegerType())) # fix age value that has trailing "_" - found in EDA
    df = df.withColumn("Age", F.when((col("Age") <= 0) | (col("Age") > 100), None).otherwise(col("Age"))) # fix age into reasonable range 0-100
    
    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_daily_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    # df.toPandas().to_parquet(filepath,
    #           compression='gzip')
    print('saved to:', filepath)
    
    return df