import os
import sys
import time

from collections import defaultdict, deque
from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task2')
sc.setLogLevel("WARN")
# spark-submit task2.py <filter threshold> <input_file_path> <betweenness_output_file_path> <community_output_file_path>


def get_betweenness(graph_input):
    # Class to represent a node in the graph
    class Node:
        def __init__(self, name):
            self.name = name
            self.level = 0
            self.node_score = 1.0
            self.edge_score = 0.0
            self.parent = {}
            self.children = {}

    # Initialize the betweenness result
    betweenness_result = defaultdict(float)

    def build_BFS(source):
        # Initialize the tree
        tree = defaultdict(set)
        tree[0].add(Node(source))
        # Initialize the processed set
        processed = {source}
        process_dict = {source: Node(source)}
        # BFS queue
        queue = deque([source])

        # BFS
        while len(queue) != 0:
            # Pop the first element
            current_node = queue.popleft()
            # Check if the node has neighbors
            for work_node in graph_input[current_node]:
                if work_node not in processed:
                    # Update the tree
                    rookie_node = Node(work_node)
                    rookie_node.level = process_dict[current_node].level + 1
                    rookie_node.parent[current_node] = process_dict[current_node]
                    rookie_node.node_score = process_dict[current_node].node_score

                    # Update the process dict
                    process_dict[current_node].children[work_node] = rookie_node
                    process_dict[work_node] = rookie_node
                    processed.add(work_node)

                    # Add the neighbor node to the queue
                    queue.append(work_node)

                    # Add the node to the tree
                    tree[rookie_node.level].add(rookie_node)

                # Check if the neighbor node is in the same level
                elif process_dict[work_node].level > process_dict[current_node].level:
                    # Update tree
                    process_dict[work_node].node_score += process_dict[current_node].node_score
                    process_dict[work_node].parent[current_node] = process_dict[current_node]
                    process_dict[current_node].children[process_dict[work_node]] = process_dict[work_node]

        return tree
    # Build the BFS tree for each node
    for node_name in graph_input.keys():
        bfs_tree = build_BFS(node_name)
        for level in range(max(bfs_tree.keys()), 0, -1):
            for node in bfs_tree[level]:
                score_sum = node.edge_score + 1
                for parent_name in node.parent.keys():
                    # Calculate the betweenness of the edge
                    edge_betweenness = (score_sum / node.node_score) * node.parent[parent_name].node_score
                    node.parent[parent_name].edge_score += edge_betweenness
                    edge = frozenset({node.name, parent_name})
                    betweenness_result[edge] += edge_betweenness

    # Divide the betweenness by 2 to get the correct result
    for k in betweenness_result:
        betweenness_result[k] /= 2

    return betweenness_result


def get_communities(graph_input, edges_input, node_degree):
    # Find the largest communities
    max_modularity, max_communities = find_modularity(graph_input, edges_input, node_degree)
    while edges_input:
        # Find the edge with the largest betweenness
        edge_between = get_betweenness(graph_input)
        edges_to_remove = find_largest_betweenness(edge_between)
        # Remove the edge with the largest betweenness
        graph_input = cut_edge(graph_input, edges_to_remove)
        edges_input.difference_update(edges_to_remove)
        # Find the modularity of the current graph
        cur_modularity, cur_communities = find_modularity(graph_input, edges_input, node_degree)
        if cur_modularity > max_modularity:
            # Update the max modularity and communities
            max_modularity = cur_modularity
            max_communities = cur_communities
    return max_communities


def find_modularity(graph_input, edges_input, node_degree):
    m = sum(node_degree.values())
    modularity = 0
    community_res = []
    node_visited = set()

    # BFS to find the communities
    while len(node_visited) < len(graph_input):
        node = next(iter(graph_input.keys() - node_visited))
        community = set()
        queue = [node]
        while queue:
            node = queue.pop(0)
            community.add(node)
            for neighbor in graph_input[node]:
                if neighbor not in community:
                    queue.append(neighbor)
        community_res.append(community)
        node_visited.update(community)

    # Calculate modularity
    for community in community_res:
        for node_x in community:
            # Calculate the modularity for each node in the community
            for node_y in community:
                # Check if the edge is in the original graph
                if frozenset({node_x, node_y}) in edges_input:
                    modularity += 1 - node_degree[node_x] * node_degree[node_y] / m
                else:
                    modularity -= node_degree[node_x] * node_degree[node_y] / m
    modularity /= m
    return modularity, community_res


def cut_edge(graph_input, targeted_edge):
    # Make a copy of the graph
    graph_copy = {node: set(neighbors) for node, neighbors in graph_input.items()}
    for edge in targeted_edge:
        node_x, node_y = tuple(edge)
        # Remove the edge from the graph copy
        if node_y in graph_copy[node_x]:
            graph_copy[node_x].discard(node_y)
        if node_x in graph_copy[node_y]:
            graph_copy[node_y].discard(node_x)
    return graph_copy


def find_largest_betweenness(edge_betweenness_value):
    if not edge_betweenness_value:
        return set()
    max_between = max(edge_betweenness_value.values())
    # Get the edges with the highest betweenness
    edge_res = {edge for edge, between in edge_betweenness_value.items() if between == max_between}
    return edge_res


if __name__ == "__main__":
    start_time = time.time()
    # Task 2 inputs
    filter_threshold = sys.argv[1]
    input_file_path = sys.argv[2]
    betweenness_output_file_path = sys.argv[3]
    community_output_file_path = sys.argv[4]

    # For local testing
    # filter_threshold = 7
    # input_file_path = "../resource/asnlib/publicdata/ub_sample_data.csv"
    # betweenness_output_file_path = "../output/task_2_betweeness.txt"
    # community_output_file_path = "../output/task_2_community.txt"

    # Read the input file
    input_rdd = sc.textFile(input_file_path)  # Add partition to avoid crashing if needed
    # Remove the header and preprocess
    header = input_rdd.first()
    input_rdd = input_rdd.filter(lambda line: line != header)\
        .map(lambda line: line.split(','))\
        .map(lambda line: (line[0], line[1]))

    # Obtain user_id to business_id map
    user_business = input_rdd.groupByKey().collectAsMap()

    # Generate user pairs
    user_pairs = [(user1, user2) for index, user1 in enumerate(list(user_business.keys()))
                  for user2 in list(user_business.keys())[index + 1:]]

    # Convert to rdd
    user_pairs_rdd = sc.parallelize(user_pairs)

    # Apply filter threshold
    filtered_user_pairs = user_pairs_rdd.filter(
        lambda line: len(set(user_business[line[0]]) & set(user_business[line[1]])) >= int(filter_threshold))

    # Generate vertices
    vertices = filtered_user_pairs.flatMap(lambda line: [(line[0],), (line[1],)]).distinct()

    # Generate edges
    edges = filtered_user_pairs.flatMap(lambda line: [line, (line[1], line[0])])

    # Create graph
    graph = edges.groupByKey()\
        .mapValues(set)\
        .collectAsMap()

    # Find the degree of nodes
    graph_degree = dict()
    for key, value in graph.items():
        graph_degree[key] = len(value)

    # Task 2_1
    # Find the betweenness of nodes
    index_betweenness = get_betweenness(graph)

    # Remap the betweenness to the original node name
    # Round the betweenness to 5 decimal places
    betweenness = dict()
    for key, value in index_betweenness.items():
        betweenness[sorted(list(key))[0], sorted(list(key))[1]] = round(value, 5)

    # Sort the betweenness in descending order then sort the user_id in lexicographical order
    sorted_betweenness = sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))

    # Betweenness write to file
    with open(betweenness_output_file_path, 'w') as f:
        for line in sorted_betweenness:
            f.write(str(line)[1:-1] + '\n')

    # Task 2_2
    # Find the communities
    communities = get_communities(graph, set(index_betweenness.keys()), graph_degree)

    # Sort the communities in descending order then sort the user_id in lexicographical order
    sorted_communities = sorted(communities, key=lambda x: (len(x), sorted(list(x))[0]))

    # Communities write to file
    with open(community_output_file_path, 'w') as f:
        for line in sorted_communities:
            f.write(str(sorted(list(line)))[1:-1] + '\n')

    # For local testing
    # print("Duration: {}".format(time.time() - start_time))
