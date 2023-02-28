import json
import os
import sys
import time

from operator import add
from pyspark.context import SparkContext


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# create Spark context with Spark configuration
sc = SparkContext('local[*]', 'task2')


if __name__ == "__main__":
    # read command line inputs (input and output file name)
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    partition_size = int(sys.argv[3])
    # input_file = '../resource/asnlib/publicdata/business.json'
    # output_file = '../resource/asnlib/publicdata/task2_output.json'
    # partition_size = 5

    # output dictionary
    result = {}

    # read input file to RDD and look for business_id
    review_file = sc.textFile(input_file).map(json.loads).map(lambda x: (x["business_id"], 1))
    # get number of partitions and items
    n_partition = review_file.getNumPartitions()
    n_items = review_file.glom().map(len).collect()

    # record start time for default partition task
    start_time = time.time()
    # find top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business = review_file.reduceByKey(add) \
        .sortBy(lambda x: [-x[1], x[0]]) \
        .map(list) \
        .take(10)
    # record task end time
    exe_time = time.time() - start_time
    result["default"] = {"n_partition": n_partition,
                         "n_items": n_items,
                         "exe_time": exe_time}

    # read input text file to RDD and repartition
    review_file_repartition = sc.textFile(input_file).map(json.loads).map(lambda x: (x["business_id"], 1))\
        .partitionBy(partition_size)
    # get number of partitions and items
    n_partition_customized = review_file_repartition.getNumPartitions()
    n_items_customized = review_file_repartition.glom().map(len).collect()

    # record start time for customized partition task
    start_time = time.time()
    # find top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business_customized = review_file_repartition.reduceByKey(add) \
        .sortBy(lambda x: [-x[1], x[0]]) \
        .map(list) \
        .take(10)
    # record task end time
    exe_time = time.time() - start_time
    result["customized"] = {"n_partition": n_partition_customized,
                            "n_items": n_items_customized,
                            "exe_time": exe_time}

    # save output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
