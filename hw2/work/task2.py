import itertools
import math
import os
import sys
import time

from collections import defaultdict, Counter
from pyspark import SparkContext


# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task2')


def count_dict(items, support_value):
    item_cnt = {item: 0 for item in set(items)}
    for item in items:
        item_cnt[item] += 1
    count_filtered = {k: v for k, v in item_cnt.items() if v >= support_value}
    return count_filtered


def count_dict_tuple(work_chunk, support_value, work_candidates):
    tuple_cnt = defaultdict(int)
    for basket in work_chunk:
        for candidate in work_candidates:
            if set(candidate).issubset(basket):
                tuple_cnt[candidate] += 1
    count_filtered = {k: v for k, v in tuple_cnt.items() if v >= support_value}
    return count_filtered


def get_frequent_candidates(count_dict_input, support_value, tuple_flag):
    frequent_candidate_list = [candidate for candidate, count in count_dict_input.items() if count >= support_value]
    # Update temp result for printing
    for candidate, count in count_dict_input.items():
        if count >= support_value:
            pass_1_temp.append((candidate if tuple_flag else tuple({candidate}), 1))
    return frequent_candidate_list


def generate_candidates(frequent_item, tuple_size):
    candidates = set()
    previous_candidates = set()
    for item_x in frequent_item:
        for item_y in frequent_item:
            if item_x != item_y and len(set(item_x).intersection(set(item_y))) == tuple_size - 2:
                new_set = tuple(sorted(set(item_x).union(set(item_y))))
                if len(new_set) == tuple_size and new_set not in candidates:
                    if not previous_candidates or all(tuple(sorted(candidate)) in frequent_item for candidate in
                                                      itertools.combinations(new_set, tuple_size - 1)):
                        candidates.add(new_set)
        previous_candidates.update(itertools.combinations(item_x, tuple_size - 1))
    return list(candidates)


def apriori(basket_input):
    task_basket = list(basket_input)
    scaled_support = support * (len(task_basket) / num_transaction)

    # Create list of all items in baskets
    items = [item for basket in task_basket for item in basket]

    # Create count dictionary for items
    item_counts = Counter(items)
    item_counts = {k: v for k, v in item_counts.items() if v >= scaled_support}

    # Find frequent item sets of k = 1
    frequent_items = get_frequent_candidates(item_counts, scaled_support, False)

    # Indicator of k item sets
    # Current step: k = 2
    # Generating pairs of frequent items
    candidates = list(itertools.combinations(sorted(frequent_items), 2))
    frequent_items_cnt = count_dict_tuple(task_basket, scaled_support, candidates)
    frequent_items = get_frequent_candidates(frequent_items_cnt, scaled_support, True)

    # Start generating frequent item sets of size 3 and above
    k = 3
    while len(frequent_items) > 0:
        candidates = generate_candidates(frequent_items, k)
        frequent_items_cnt = count_dict_tuple(task_basket, scaled_support, candidates)
        frequent_items = get_frequent_candidates(frequent_items_cnt, scaled_support, True)
        k += 1
    return [pass_1_temp]


def find_frequent_itemsets(candidate, support_value, transaction_list):
    frequent_item_count = defaultdict(int)
    for transaction in transaction_list:
        if all(item in transaction for item in candidate):
            frequent_item_count[candidate] += 1

    frequent_itemsets = [(tuple(sorted(itemset)), count) for itemset, count in frequent_item_count.items()
                         if count >= support_value]
    return frequent_itemsets


def format_string(input_tuple, size, output):
    if len(input_tuple) > size:
        return output + "\n\n"

    if len(input_tuple) == 1:
        output += f"('{input_tuple[0]}'),"
    else:
        output += f"{input_tuple},"

    return output


if __name__ == "__main__":
    start_time = time.time()
    filter_threshold = int(sys.argv[1])
    support = int(sys.argv[2])
    input_file_path = sys.argv[3]
    output_file_path = sys.argv[4]

    # For testing locally
    # filter_threshold = 20
    # support = 50
    # input_file_path = "../resource/asnlib/publicdata/ta_feng_all_months_merged.csv"
    # output_file_path = "output_task2.txt"

    # Preprocessing
    raw_rdd = sc.textFile(input_file_path)  # Limit partitions to avoid crashing
    header = raw_rdd.first()
    preprocess_rdd = raw_rdd.filter(lambda x: x != header)\
        .map(lambda x: x.split(","))\
        .map(lambda x: (x[0].strip('"'), x[1].strip('"'), int(x[5].strip('"'))))
    preprocess_rdd = preprocess_rdd.collect()

    # Save intermediate CSV file
    intermediate_csv = "customer_product.csv"
    with open(intermediate_csv, "w") as f:
        f.write("DATE-CUSTOMER_ID,PRODUCT_ID\n")
        for x in preprocess_rdd:
            f.write(x[0][:-2] + "-" + x[1] + "," + str(x[2]) + "\n")

    # Read intermediate CSV file for future tasks
    new_ta_feng_rdd = sc.textFile(intermediate_csv)
    new_header = new_ta_feng_rdd.first()
    new_ta_feng_rdd = new_ta_feng_rdd.filter(lambda x: x != new_header)\
        .map(lambda x: x.split(","))\
        .map(lambda x: (x[0], x[1]))\
        .groupByKey()\
        .mapValues(set)\
        .filter(lambda x: len(x[1]) > filter_threshold)\
        .map(lambda x: x[1])\
        .persist()

    # Pass into pipeline similar to task1
    basket_list = new_ta_feng_rdd.collect()
    num_transaction = len(basket_list)
    pass_1_temp = list()
    pass_1_mapper = new_ta_feng_rdd.mapPartitions(apriori).flatMap(lambda x: x)
    pass_1_reducer = pass_1_mapper.reduceByKey(lambda x, y: x + y)

    pass_2_mapper = pass_1_reducer.map(lambda x: find_frequent_itemsets(x[0], support, basket_list)).flatMap(lambda x: x)
    pass_2_reducer = pass_2_mapper.filter(lambda x: x[1] >= support).map(lambda x: x[0]).collect()

    pass_1_final = ""
    pass_1_result = sorted(sorted(pass_1_reducer.map(lambda x: x[0]).collect()),
                           key=lambda element: (len(element), element))
    size = len(pass_1_result[0])
    for x in pass_1_result:
        if len(x) != size:
            pass_1_final = pass_1_final[:-1] + "\n\n"
            size = len(x)
        pass_1_final = format_string(x, size, pass_1_final)

    pass_2_final = ""
    pass_2_result = sorted(sorted(pass_2_reducer),
                           key=lambda element: (len(element), element))
    size = len(pass_2_result[0])
    for x in pass_2_result:
        if len(x) != size:
            pass_2_final = pass_2_final[:-1] + "\n\n"
            size = len(x)
        pass_2_final = format_string(x, size, pass_2_final)

    with open(output_file_path, 'w') as file:
        file.write("Candidates:\n" + pass_1_final[:-1] + "\n\nFrequent Itemsets:\n" + pass_2_final[:-1])

    print("Duration:", time.time() - start_time)
