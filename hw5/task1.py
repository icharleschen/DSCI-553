import os
import sys
import time
import binascii

from blackbox import BlackBox
# from pyspark import SparkContext

# Spark configuration
# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# sc = SparkContext('local[*]', 'task1')
# sc.setLogLevel("WARN")


# Define hash functions
# f(x) = ((ax + b) % p) % m
def hash_function():
    a = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
    b = [i+1 for i in a]
    p = [88799, 60961, 74363, 17117, 68111, 68071, 48883, 37889, 90011, 49807]
    m = 69997
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


# Implement bloom filter and calculate false positive rate
def bloom_filter(user_stream):
    # Initialize false positive and true negative trackers
    fp = 0
    tn = 0

    # Iterate through user stream
    for user in user_stream:
        # Apply hash functions on each user
        hash_values = myhashs(user)
        user_tracker = 0

        # Check if user is in filter bit array
        for hash_value in hash_values:
            if filter_bit_array[hash_value] == 0:
                filter_bit_array[hash_value] = 1
            else:
                user_tracker += 1
        if user not in previous_user:
            if user_tracker == 10:  # Size of hash_values
                fp += 1
            else:
                tn += 1

        # Update previous user list
        previous_user.add(user)

    # Calculate false positive rate
    fpr = fp / (fp + tn)
    return fpr


if __name__ == '__main__':
    start_time = time.time()
    # Task 1 inputs
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    # Local testing filepaths
    # input_filename = "../resource/asnlib/publicdata/users.txt"
    # stream_size = int('300')
    # num_of_asks = int('30')
    # output_filename = "../output/task1_result.txt"

    # Initialize global filter bit array
    filter_bit_array = [0 for i in range(69997)]
    # print(len(filter_bit_array))

    # Initialize bloom filter result list
    bloom_filter_result = list()
    # Initialize previous user list globally
    previous_user = set()

    # Ask blackbox multiple times
    bx = BlackBox()
    for _ in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        # Apply bloom filter on user stream
        bloom_filter_result.append(bloom_filter(stream_users))

    # Write result to output file
    with open(output_filename, 'w') as f:
        f.write("Time,FPR\n")
        for index in range(num_of_asks):
            f.write("{},{}\n".format(index, bloom_filter_result[index]))

    # Run time
    print("Duration: {}".format(time.time() - start_time))
