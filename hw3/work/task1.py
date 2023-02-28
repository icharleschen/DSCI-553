import os
import sys
from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task1')


if __name__ == "__main__":
    # Handle input arguments
    # input_file_name = sys.argv[1]
    # output_file_name = sys.argv[2]

    # For local testing
    input_file_name = "../asnlib/publicdata/yelp_train.csv"
    output_file_name = "../task1_output.csv"

    # Read csv into RDD
    rdd = sc.textFile(input_file_name, 200)  # Add number of partitions to avoid crashing
    header = rdd.first()
    rdd = rdd.filter(lambda x: x != header)  # Remove header

    print(rdd.take(5))
