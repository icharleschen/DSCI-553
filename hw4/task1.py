import os
import sys
import time

from graphframes import *
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
sc = SparkContext('local[*]', 'task1')
sqlContext = SQLContext(sc)


if __name__ == "__main__":
    start_time = time.time()
    # Task 1 inputs
    filter_threshold = sys.argv[1]
    input_file_path = sys.argv[2]
    community_file_path = sys.argv[3]

    # Read the input file
    input_rdd = sc.textFile(input_file_path)
    # Remove the header
    header = input_rdd.first()
    input_rdd = input_rdd.filter(lambda line: line != header)\
        .map(lambda line: line.split(','))

    # For local testing
    print("Duration: {}", time.time() - start_time)
