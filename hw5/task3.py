import os
import sys
import time
import random


from blackbox import BlackBox
# from pyspark import SparkContext


# Spark configuration
# os.environ['PYSPARK_PYTHON'] = sys.executable
# os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# sc = SparkContext('local[*]', 'task3')
# sc.setLogLevel("WARN")


# Implement Fixed Size Sampling
def fixed_size_sampling(user_stream, num_ask):
    global all_users, sequence_number
    # For the first ask, save all users in the stream
    if num_ask == 0:
        for i in range(len(user_stream)):
            all_users[i] = user_stream[i]
            sequence_number += 1

    # For rest of asks, randomly replace users in the stream
    # If the randomly generated probability is less than s/n, we accept the sample
    # Pick a random user from the stream and replace it with a user from the stream
    else:
        for user in user_stream:
            if random.random() < (stream_size / sequence_number):
                all_users[random.randint(0, memory_size - 1)] = user
            sequence_number += 1

    # Return sequence number, 1, 21, 41, 61, and 81 users in all users
    return [sequence_number-1, all_users[0], all_users[20], all_users[40], all_users[60], all_users[80]]


if __name__ == '__main__':
    start_time = time.time()
    # Task 3 input
    input_filename = sys.argv[1]
    stream_size = int(sys.argv[2])
    num_of_asks = int(sys.argv[3])
    output_filename = sys.argv[4]

    # Local testing filepaths
    # input_filename = "../resource/asnlib/publicdata/users.txt"
    # stream_size = int('100')
    # num_of_asks = int('30')
    # output_filename = "../output/task3_result.csv"

    # Set random seed
    random.seed(553)

    # Set memory size according to HW instructions
    memory_size = 100
    # Initialize a list to store all users
    all_users = [0 for i in range(memory_size)]
    # Initialize sequence number to keep track of asks
    sequence_number = 1

    # Store result for all asks
    fixed_size_sampling_result = list()

    # Ask blackbox multiple times
    bx = BlackBox()
    for ask in range(num_of_asks):
        stream_users = bx.ask(input_filename, stream_size)
        fixed_size_sampling_result.append(fixed_size_sampling(stream_users, ask))

    # print(fixed_size_sampling_result)

    # Write result to output file
    with open(output_filename, 'w') as f:
        f.write("seqnum,0_id,20_id,40_id,60_id,80_id\n")
        for index in range(len(fixed_size_sampling_result)):
            f.write("{},{},{},{},{},{}\n".format(fixed_size_sampling_result[index][0],
                                                 fixed_size_sampling_result[index][1],
                                                 fixed_size_sampling_result[index][2],
                                                 fixed_size_sampling_result[index][3],
                                                 fixed_size_sampling_result[index][4],
                                                 fixed_size_sampling_result[index][5]))

    # Run time
    # print("Duration: {}".format(time.time() - start_time))
