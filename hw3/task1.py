import itertools
import os
import random
import sys
import time

from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task1')

# Set hash function parameters
num_bands = 50
num_hash = 100
num_rows = num_hash // num_bands

# Jaccard threshold
jaccard_threshold = 0.5

# random.seed(42)


# Generate minhash function parameters
def generate_minhash_para(hash_size):
    # Randomly generate unique a and b for hash function matching size
    # f(x) = (ax + b) % m
    a_para = random.sample(range(1, 250), hash_size)
    b_para = random.sample(range(1, 250), hash_size)
    minhash_para = [[a_para[i], b_para[i]] for i in range(hash_size)]
    return minhash_para


# Define function to find signature matrix using minhash values
def minhash(data_input, num_user, hash_para):
    # Iteratively find minhash value for each hash function
    hash_result = list()
    m = num_user
    for i in range(num_hash - 1):
        a = hash_para[i][0]
        b = hash_para[i][1]
        # print("a: {}, b: {}".format(a, b))
        hash_result.append(min([(a * x + b) % m for x in data_input]))
    return hash_result


def LSH(business, users, num_band, num_row):
    band_split = []
    for band in range(num_band):
        users_range = [band * num_row, (band + 1) * num_row]
        band_tuple = (band, tuple(users[users_range[0]:users_range[1]]))
        band_split.append((band_tuple, [business]))
    return band_split


def jaccard_similarity(candidate1, candidate2):
    candidate_set1 = set(chara_matrix_dict[candidate1])
    candidate_set2 = set(chara_matrix_dict[candidate2])
    intersection = len(candidate_set1.intersection(candidate_set2))
    union = len(candidate_set1) + len(candidate_set2) - intersection
    return intersection / union, candidate1, candidate2


if __name__ == "__main__":
    start_time = time.time()
    # Handle input arguments
    input_file_name = sys.argv[1]
    output_file_name = sys.argv[2]

    # For local testing
    # input_file_name = "../asnlib/publicdata/yelp_train.csv"
    # output_file_name = "task1_output.csv"

    # Read csv into RDD and preprocess
    input_rdd = sc.textFile(input_file_name)  # Consider add number of partitions to avoid crashing
    header = input_rdd.first()  # Remove header
    raw_rdd = input_rdd.filter(lambda x: x != header)\
        .map(lambda x: x.split(','))\
        .map(lambda x: (x[0], x[1]))

    # Find and collect users
    # ("user_id": int(user_index))
    user_rdd = raw_rdd.map(lambda x: x[0])\
        .distinct()\
        .sortBy(lambda x: x)\
        .zipWithIndex()
    user_rdd_dict = sc.broadcast(user_rdd.collectAsMap())
    num_users = user_rdd.count()
    # print(user_rdd.take(10))
    # print("Number of users: {}".format(num_users))

    # Find and collect businesses
    # ("business_id": int(business_index))
    business_rdd = raw_rdd.map(lambda x: x[1])\
        .distinct()\
        .sortBy(lambda x: x)\
        .zipWithIndex()
    business_rdd_dict = sc.broadcast(business_rdd.collectAsMap())
    num_businesses = business_rdd.count()
    # print(business_rdd.take(10))
    # print("Number of businesses: {}".format(num_businesses))

    # Find characteristic matrix
    # ("business_id": [user_index1, user_index2, ...])
    chara_matrix = raw_rdd.map(lambda x: (x[1], user_rdd_dict.value[x[0]]))\
        .groupByKey()\
        .map(lambda x: (x[0], list(x[1])))\
        .sortBy(lambda x: x[0])
    # print(chara_matrix.take(10))
    chara_matrix_dict = chara_matrix.collectAsMap()

    # Obtain minhash values
    minhash_list = generate_minhash_para(num_hash)
    # print(minhash_list[:10])

    # Find signature matrix
    signature_matrix = chara_matrix.map(lambda x: (x[0], minhash(x[1], num_users, minhash_list)))
    # print(signature_matrix.take(5))

    # Apply LSH
    LSH_result = signature_matrix.flatMap(lambda x: LSH(x[0], x[1], num_bands, num_rows))\
        .reduceByKey(lambda x, y: x + y).filter(lambda x: len(x[1]) > 1)
    # print(LSH_result.take(5))
    # print(LSH_result.count())

    # Find candidate pairs
    candidate_pairs = LSH_result.flatMap(lambda x: sorted(itertools.combinations(sorted(x[1]), 2))).distinct()
    # print(candidate_pairs.take(5))

    # Compute Jaccard similarity and filter on threshold
    jaccard_result = candidate_pairs.map(lambda x: jaccard_similarity(x[0], x[1]))\
        .filter(lambda x: x[0] >= jaccard_threshold)\
        .sortBy(lambda x: (x[1], x[2]))\
        .sortBy(lambda x: x[1])
    # print(jaccard_result.take(5))

    # Save output
    with open(output_file_name, 'w') as f:
        f.write("business_id_1, business_id_2, similarity\n")
        for line in jaccard_result.collect():
            f.write("{},{},{}\n".format(str(line[1]), str(line[2]), str(line[0])))

    # For local testing
    # print("Duration: {}".format(time.time() - start_time))
