import os
import sys
import time

from graphframes import *
from pyspark import SparkContext
from pyspark.sql import SQLContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
# os.environ["PYSPARK_SUBMIT_ARGS"] = "--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell"
sc = SparkContext('local[*]', 'task1')
sc.setLogLevel("WARN")
sqlContext = SQLContext(sc)
# spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py <filter threshold> <input_file_path> <community_output_file_path>
# /opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py 7 ../resource/asnlib/publicdata/ub_sample_data.csv task_1.txt


if __name__ == "__main__":
    start_time = time.time()
    # Task 1 inputs
    # filter_threshold = sys.argv[1]
    # input_file_path = sys.argv[2]
    # community_file_path = sys.argv[3]

    # For local testing
    filter_threshold = 7
    input_file_path = "../resource/asnlib/publicdata/ub_sample_data.csv"
    community_file_path = "task_1output.txt"

    # Read the input file
    input_rdd = sc.textFile(input_file_path)  # Add partition to avoid crashing if needed
    # Remove the header and preprocess
    header = input_rdd.first()
    input_rdd = input_rdd.filter(lambda line: line != header)\
        .map(lambda line: line.split(','))\
        .map(lambda line: (line[0], line[1]))

    # Obtain user_id to business_id map
    user_business = input_rdd.groupByKey().collectAsMap()
    # print(user_business)

    # Generate user pairs
    user_pairs = [(user1, user2) for index, user1 in enumerate(list(user_business.keys()))
                  for user2 in list(user_business.keys())[index + 1:]]
    # print(len(user_pairs))

    # Convert to rdd
    user_pairs_rdd = sc.parallelize(user_pairs)

    # Apply filter threshold
    filtered_user_pairs = user_pairs_rdd.filter(
        lambda line: len(set(user_business[line[0]]) & set(user_business[line[1]])) >= filter_threshold)
    # print(filtered_user_pairs.count())

    # Generate vertices
    vertices = filtered_user_pairs.flatMap(lambda line: [(line[0],), (line[1],)]).distinct()
    print("vertices count:", vertices.count())
    # Generate edges
    edges = filtered_user_pairs.map(lambda line: (line[0], line[1]))
    print("edges count:", edges.count())

    # Convert to dataframes
    vertices_df = sqlContext.createDataFrame(vertices, ['id'])
    edges_df = sqlContext.createDataFrame(edges, ['src', 'dst'])
    print((vertices_df.count(), len(vertices_df.columns)))
    print((edges_df.count(), len(edges_df.columns)))

    # Create graph
    graph = GraphFrame(vertices_df, edges_df)
    # Label Propagation Algorithm (LPA)
    communities = graph.labelPropagation(maxIter=5)

    # Process output rdd
    result_rdd = communities.rdd.map(lambda line: (line[1], line[0]))\
        .groupByKey()\
        .map(lambda label: sorted(list(label[1])))\
        .sortBy(lambda line: (len(line), line[0]))

    # Write to file
    with open(community_file_path, 'w') as f:
        for line in result_rdd.collect():
            f.write(str(line)[1:-1] + '\n')

    # For local testing
    print("Duration: {}".format(time.time() - start_time))
