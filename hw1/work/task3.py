import json
import os
import sys
import time

from pyspark.context import SparkContext


os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# create Spark context with Spark configuration
sc = SparkContext('local[*]', 'task3')


def LoadFile(review_path, business_path):
    # read input JSON files to RDD
    review_raw = sc.textFile(review_path).map(json.loads)
    business_raw = sc.textFile(business_path).map(json.loads)
    # discard records with empty “city” field
    business_clean = business_raw.filter(lambda x: x["city"] != "null")
    return review_raw, business_clean


def AverageStars(review_rdd, business_rdd):
    # find business average stars by using review_file
    business_avg_stars = review_rdd.map(lambda x: (x["business_id"], (x["stars"], 1))) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda x: x[0] / x[1])
    # produce business_id: city key-value paris
    business_city = business_rdd.map(lambda x: (x["business_id"], x["city"]))
    # merge rdd, change key-value pairs format, and calculate average stars by city
    business_city_stars = business_city.join(business_avg_stars) \
        .map(lambda a: (a[1][0], (a[1][1], 1))) \
        .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
        .mapValues(lambda x: x[0] / x[1])
    return business_city_stars


if __name__ == "__main__":
    # read command line inputs (input and output file name)
    review_filepath = sys.argv[1]
    business_filepath = sys.argv[2]
    output_filepath_question_a = sys.argv[3]
    output_filepath_question_b = sys.argv[4]
    # review_filepath = '../resource/asnlib/publicdata/test_review.json'
    # business_filepath = '../resource/asnlib/publicdata/business.json'
    # output_filepath_question_a = '../resource/asnlib/publicdata/task3_a.txt'
    # output_filepath_question_b = '../resource/asnlib/publicdata/task3_b.json'

    # output part b
    result_b = {}

    # record start time for M1 task
    start_time = time.time()
    # call function to obtain data from device
    review_file, business_file = LoadFile(review_filepath, business_filepath)
    # call function to get city and average star rdd
    city_average_stars = AverageStars(review_file, business_file)
    # sort using Python
    city_average_stars = sorted(city_average_stars.collect(), key=lambda x: [-x[1], x[0]])
    print(city_average_stars[:10])
    M1_time = time.time() - start_time

    # record start time for M2 task
    start_time = time.time()
    # call function to obtain data from device
    review_file, business_file = LoadFile(review_filepath, business_filepath)
    # call function to get city and average star rdd
    city_average_stars = AverageStars(review_file, business_file)
    # sort using rdd
    city_average_stars = city_average_stars.sortBy(lambda x: [-x[1], x[0]])
    print(city_average_stars.collect()[:10])
    M2_time = time.time() - start_time

    # format part b output
    result_b["m1"] = M1_time
    result_b["m2"] = M2_time
    result_b["reason"] = "From the M1 and M2 time listed above, we can see that it requires more execution time " \
                         "sorting the result RDD using Python than that of using the RDD sortBy(). This this case, " \
                         "it seems like Python's sort is done on single core while spark RDD is doing it in parellel " \
                         "setting."

    # save part a text file
    with open(output_filepath_question_a, 'w') as f:
        f.write("city,stars\n")
        for line in city_average_stars.collect():
            # print(line[0], line[1])
            f.write(line[0] + "," + str(line[1]) + "\n")

    # save part b JSON
    with open(output_filepath_question_b, 'w', encoding='utf-8') as f:
        json.dump(result_b, f, indent=4)
