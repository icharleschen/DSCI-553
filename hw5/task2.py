import os
import sys
import time
import binascii

from blackbox import BlackBox
# from pyspark import SparkContext
from statistics import mean, median

# Spark configuration
# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# sc = SparkContext('local[*]', 'task2')
# sc.setLogLevel("WARN")


# Define hash functions
# f(x) = ((ax + b) % p) % m
def hash_function():
    a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    b = [i+1 for i in a]
    p = [1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069, 1087, 1091]
    m = 2**len(p)
    hash_function_list = list()
    for i in range(len(a)):
        hash_function_list.append([a[i], b[i], p[i], m])
    return hash_function_list


# Encapsulate hash functions into a function called myhashs
# Copy from the HW instruction
def myhashs(s):
    result = []
    # Convert string to int
    s = int(binascii.hexlify(s.encode('utf8')), 16)
    hash_function_list = hash_function()
    # Apply hash functions
    for f in hash_function_list:
        result.append(((f[0] * s + f[1]) % f[2]) % f[3])
    return result


# Get the number of trailing zeros in a binary number
def count_trailing_zeros(binary_value):
    # Initialize counter
    counter = 0
    # Iterate through binary value
    for i in range(len(binary_value)-1, -1, -1):
        # Check if binary value is 0
        if binary_value[i] == '0':
            # Increment counter
            counter += 1
        else:
            break
    return counter


# Get the estimated number of distinct users by
# Partition hash functions into small groups
# Take average for each group
# Take the median of the averages
def get_estimated_distinct_users(max_trailing_zeros):
    # Set group size
    group_size = 5
    # Initialize list to store averages
    averages = list()
    # Partition hash functions into small groups
    for i in range(0, len(max_trailing_zeros), group_size):
        # Get average for each group
        averages.append(mean(max_trailing_zeros[i:i+group_size]))
    # Take the median of the averages
    return median(averages)


# Implement Flajolet-Martin algorithm
def Flajolet_Martin(user_stream):
    global estimated_result, actual_result
    # Initialize list to store hash values and estimated values
    all_hash_values = list()
    estimated_values = list()

    # Iterate through user stream
    for user in user_stream:
        # Apply hash functions on each user
        hash_values = myhashs(user)
        all_hash_values.append(hash_values)

    # Iterate through hash values
    for hash_value_index in range(len(all_hash_values[0])):
        max_trailing_zeros = 0
        for hash_value in all_hash_values:
            # Get the number of trailing zeros in a binary number
            binary_value = bin(hash_value[hash_value_index])[2:]
            trailing_zeros = count_trailing_zeros(binary_value)
            if trailing_zeros > max_trailing_zeros:
                max_trailing_zeros = trailing_zeros

        # Store estimated values
        estimated_values.append(2**max_trailing_zeros)

    # Get estimated number of distinct users
    estimated_distinct_users = int(get_estimated_distinct_users(estimated_values))
    # Get actual number of distinct users
    actual_distinct_users = len(set(user_stream))

    # Add result to trackers for testing
    estimated_result += estimated_distinct_users
    actual_result += actual_distinct_users

    return [actual_distinct_users, estimated_distinct_users]


if __name__ == '__main__':
    start_time = time.time()
    # Task 2 inputs
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    # Local testing filepaths
    # input_filename = "../resource/asnlib/publicdata/users.txt"
    # stream_size = int('100')
    # num_of_asks = int('30')
    # output_filename = "../output/task2_result.csv"

    # Store result for all asks
    flajolet_martine_result = list()

    # Initialize estimated result and actual result tracker
    estimated_result = 0
    actual_result = 0

    # Ask blackbox multiple times
    bx = BlackBox()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        # Apply Flajolet-Martin algorithm on user stream
        flajolet_martine_result.append(Flajolet_Martin(stream_users))

    # Print result
    # print(flajolet_martine_result)
    # Print estimations/ground truth
    # print("Ratio {}".format(estimated_result/actual_result))

    # Write result to output file
    with open(output_filename, 'w') as f:
        f.write("Time,Ground Truth,Estimation\n")
        for index in range(num_of_asks):
            f.write("{},{},{}\n".format(index, flajolet_martine_result[index][0], flajolet_martine_result[index][1]))

    # Run time
    # print("Duration: {}".format(time.time() - start_time))
