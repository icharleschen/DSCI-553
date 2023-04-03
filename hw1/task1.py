import json
import os
import sys

from operator import add
from pyspark.context import SparkContext


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# create Spark context with Spark configuration
sc = SparkContext('local[*]', 'task1')


if __name__ == "__main__":
    # read command line inputs (input and output file name)
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # output dictionary
    result = {}

    # read input file to RDD
    review_file = sc.textFile(input_file).map(json.loads)

    # get total number of reviews
    n_review = review_file.count()
    result["n_review"] = n_review
    # print(n_review)

    # filter reviews in 2018
    n_review_2018 = review_file.filter(lambda x: x["date"].startswith('2018')).count()
    result["n_review_2018"] = n_review_2018
    # print(n_review_2018)

    # find the number of distinct users who wrote reviews
    n_user = review_file.map(lambda x: x["user_id"]).distinct().count()
    result["n_user"] = n_user
    # print(n_user)

    # find top 10 users who wrote the largest numbers of reviews and the number of reviews they wrote
    top10_user = review_file.map(lambda x: (x["user_id"], 1))\
        .reduceByKey(add)\
        .sortBy(lambda x: [-x[1], x[0]])\
        .map(list)\
        .take(10)
    result["top10_user"] = top10_user
    # print(top10_user)

    # find the number of distinct businesses that have been reviewed
    n_business = review_file.map(lambda x: x["business_id"]).distinct().count()
    result["n_business"] = n_business
    # print(n_business)

    # find top 10 businesses that had the largest numbers of reviews and the number of reviews they had
    top10_business = review_file.map(lambda x: (x["business_id"], 1)) \
        .reduceByKey(add) \
        .sortBy(lambda x: [-x[1], x[0]]) \
        .map(list) \
        .take(10)
    result["top10_business"] = top10_business
    # print(top10_business)

    # save output JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=4)
