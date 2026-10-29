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


def process_bronze_table(snapshot_date_str, d, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")
    
    # connect to source back end - IRL connect to back end source system
    #csv_file_path = "data/lms_loan_daily.csv" # previous code for one dataset
    if d == "datamart/bronze/lms/":
        csv_file_path = "data/lms_loan_daily.csv"
    elif d == "datamart/bronze/clickstream/":
        csv_file_path = "data/feature_clickstream.csv"
    elif d == "datamart/bronze/attributes/":
        csv_file_path = "data/features_attributes.csv"
    else:
        csv_file_path = "data/features_financials.csv"

    # load data - IRL ingest from back end source system
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True).filter(col('snapshot_date') == snapshot_date)
    print(snapshot_date_str + 'row count:', df.count())
    
    # save bronze table to datamart - IRL connect to database to write
    #partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv' #previous code for one dataset
    if d == "datamart/bronze/lms/":
        partition_name = "bronze_loan_daily_" + snapshot_date_str.replace('-','_') + '.csv'
    elif d == "datamart/bronze/clickstream/":
        partition_name = "bronze_clickstream_daily_" + snapshot_date_str.replace('-','_') + '.csv' 
    elif d == "datamart/bronze/attributes/":
        partition_name = "bronze_attributes_daily_" + snapshot_date_str.replace('-','_') + '.csv' 
    else:
        partition_name = "bronze_financials_daily_" + snapshot_date_str.replace('-','_') + '.csv' 
    filepath = d + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df
