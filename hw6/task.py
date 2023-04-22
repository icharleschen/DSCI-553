import collections
import math
import numpy as np
import random
import sys
import time

from sklearn.cluster import KMeans


def preprocess_data(data_partitioned, pt_ctr, pt_map, cl_map):
    data_out = [[float(i) for i in line.split(',')[2:]] for line in data_partitioned]
    for j, line in enumerate(data_partitioned):
        data_index = line.split(',')[0]
        cl_map[j + pt_ctr] = data_index
        pt_map[str(data_out[j])] = data_index
    pt_ctr += len(data_partitioned)
    return data_out, pt_ctr, pt_map, cl_map


def get_cluster_indices(cluster_data):
    # Return a dictionary of cluster indices
    work_cluster_index = {}
    for index, cluster_data_id in enumerate(cluster_data):
        if cluster_data_id not in work_cluster_index:
            work_cluster_index[cluster_data_id] = []
        work_cluster_index[cluster_data_id].append(index)
    return work_cluster_index


def get_compression_set(work_set, cluster_index, point_indices, data_points, rs_dict, rs_points):
    work_set[cluster_index][0] = []

    # Convert point indices to point IDs
    compression_set_ids = [list(rs_dict.keys())[list(rs_dict.values()).index(rs_points[i])] for i in point_indices]

    # Add compression set IDs to work_set
    work_set[cluster_index][0] = compression_set_ids

    # Calculate the number of points in the compression set
    n_points = len(point_indices)
    work_set[cluster_index][1] = n_points

    # Calculate the sum, sum of squares, and standard deviation of the points in the compression set
    compression_set_candidates = data_points[point_indices, :]
    sum_of_points = np.sum(compression_set_candidates, axis=0)
    sum_of_squares = np.sum(compression_set_candidates ** 2, axis=0)
    standard_deviation = np.sqrt((sum_of_squares / n_points) - (np.square(sum_of_points) / (n_points ** 2)))
    work_set[cluster_index][2] = sum_of_points
    work_set[cluster_index][3] = sum_of_squares
    work_set[cluster_index][4] = standard_deviation

    # Calculate the centroid of the points in the compression set
    centroid_result = np.mean(compression_set_candidates, axis=0)
    work_set[cluster_index][5] = centroid_result


def write_intermediate_result(file_name, num_round, ds, cs, rs):
    ds_cnt = sum([v[1] for v in ds.values()])
    cs_cluster_cnt = len(cs)
    cs_pts_cnt = sum([v[1] for v in cs.values()])
    rs_cnt = len(rs)

    file_name.write(f"Round {num_round}: {ds_cnt},{cs_cluster_cnt},{cs_pts_cnt},{rs_cnt}\n")


def find_closest_cluster(data_pts, work_cluster):
    min_distance = THRESHOLD
    nearest_cluster_id = -1
    for cluster_id, work_cluster in work_cluster.items():
        std_dev = work_cluster[4].astype(np.float)
        centroid = work_cluster[5].astype(np.float)
        # Mahalanobis distance
        distance = np.sqrt(np.sum(((data_pts - centroid) / std_dev) ** 2))
        if distance < min_distance:
            min_distance = distance
            nearest_cluster_id = cluster_id
    return nearest_cluster_id


def update_cluster_stats(cluster_stats, index, data_pts, cluster_id):
    cluster_data = cluster_stats[cluster_id]
    cluster_data[0].append(index)
    cluster_data[1] += 1
    for i in range(DIMENSION):
        cluster_data[2][i] += data_pts[i]
        cluster_data[3][i] += data_pts[i] ** 2
    std_dev = np.sqrt((cluster_data[3] / cluster_data[1]) - (np.square(cluster_data[2]) / (cluster_data[1] ** 2)))
    centroid = cluster_data[2] / cluster_data[1]
    cluster_stats[cluster_id] = [cluster_data[0], cluster_data[1], cluster_data[2], cluster_data[3], std_dev, centroid]


def get_nearest_cluster(data_pts_1, data_pts_2):
    nearest_clusters = dict()
    # Find the nearest cluster for each cluster in data_pts_1
    for key1, (indices1, count1, sum1, sum_squares1, stddev1, centroid1) in data_pts_1.items():
        current_nearest_cluster = key1
        min_distance = THRESHOLD

        for key2, (_, _, _, _, stddev2, centroid2) in data_pts_2.items():
            # Check if the clusters are not the same
            if key1 != key2:
                if isinstance(stddev1, np.ndarray) and isinstance(stddev2, np.ndarray):
                    diff = centroid1 - centroid2
                    cov_sum = np.diag(stddev1 ** 2) + np.diag(stddev2 ** 2)
                    cov_sum_inv = np.linalg.inv(cov_sum)
                    mahalanobis_distance = np.sqrt(np.dot(np.dot(diff, cov_sum_inv), diff))
                    # Check if the distance is less than the threshold
                    if mahalanobis_distance < min_distance:
                        min_distance = mahalanobis_distance
                        current_nearest_cluster = key2
        nearest_clusters[key1] = current_nearest_cluster
    return nearest_clusters


def merge_cluster(compression_set, cs1_key, cs2_key):
    cs1 = compression_set[cs1_key]
    cs2 = compression_set[cs2_key]

    cs1[0].extend(cs2[0])  # merge the compressed vectors
    cs1[1] += cs2[1]  # update the count

    for i in range(DIMENSION):
        cs1[2][i] += cs2[2][i]  # update the sum of values
        cs1[3][i] += cs2[3][i]  # update the sum of squares

    # update the standard deviation and mean
    mean = np.divide(cs1[2], cs1[1])
    variance = np.divide(cs1[3], cs1[1]) - np.square(mean)
    std_dev = np.sqrt(variance)
    cs1[4] = std_dev
    cs1[5] = mean

    # update the compression set with the merged values
    compression_set[cs1_key] = cs1
    del compression_set[cs2_key]


if __name__ == '__main__':
    start_time = time.time()
    # Handle the input arguments
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    # Local file path
    # input_file = '../resource/asnlib/publicdata/hw6_clustering.txt'
    # n_cluster = 10
    # output_file = '../output/task_output.txt'

    # Load data as a numpy array
    f = open(input_file, 'r')
    data_raw = np.array(f.readlines())
    f.close()

    # Get the size of the data and 20% of the data
    random.seed(553)
    data_size = data_raw.shape[0]
    TWENTY_PERCENT = math.ceil(data_size * 0.2)
    end_index = TWENTY_PERCENT

    # STEP 1: Load 20% of the data randomly
    np.random.shuffle(data_raw)
    first_pass = data_raw[:end_index]

    # Define the data structure to store the data
    # cluster_index_map: {index of the data point in the model_feed: index of the data point in the original data}
    # point_index_map: {data point in the model_feed: index of the data point in the original data}
    # point_ctr: the number of data points in the model_feed
    cluster_index_map = {}
    point_index_map = {}
    point_ctr = 0

    # Prepare the data for K-Means
    model_feed, point_ctr, point_index_map, cluster_index_map = preprocess_data(first_pass, point_ctr,
                                                                                point_index_map, cluster_index_map)

    # Obtain dimension of the data
    DIMENSION = len(model_feed[0])
    THRESHOLD = 2 * math.sqrt(DIMENSION)

    # STEP 2: Run k-means clustering with large K, e.g. 5 times the number of clusters
    # Use the Euclidean distance as the similarity measurement
    k_means = KMeans(n_clusters=n_cluster * 5, random_state=553)
    cluster_result = k_means.fit_predict(np.array(model_feed))

    # Keep track of the clusters
    cluster_record = collections.defaultdict(list)
    for i, cluster_id in enumerate(cluster_result):
        cluster_record[cluster_id].append(model_feed[i])

    # STEP 3: Move all the clusters that contain only one point to RS (outliers)
    retained_set = {}
    removed_indices = set()
    for i, cluster_points in enumerate(cluster_record.values()):
        if len(cluster_points) == 1:
            point = cluster_points[0]
            cluster_index = list(cluster_record.keys())[i]
            retained_set[cluster_index] = point
            removed_indices.add(model_feed.index(point))
    model_feed = [point for i, point in enumerate(model_feed) if i not in removed_indices]

    # STEP 4: Run K-Means again to cluster the rest of the data points with K = the number of input clusters
    k_means = KMeans(n_clusters=n_cluster, random_state=553)
    cluster_result = k_means.fit_predict(np.array(model_feed))
    cluster_index_result = get_cluster_indices(cluster_result)

    # STEP 5: Use the K-Means result from Step 4 to generate the DS clusters
    # (i.e., discard their points and generate statistics)
    discard_set = dict()
    for key in cluster_index_result.keys():
        discard_set[key] = {
            # Data points
            0: [cluster_index_map[i] for i in cluster_index_result[key]],
            # Number of data points (N)
            1: len(cluster_index_result[key]),
            # Sum of data points (SUM)
            2: np.sum(np.array(model_feed)[cluster_index_result[key]], axis=0),
            # Sum of square of data points (SUMSQ)
            3: np.sum(np.square(np.array(model_feed)[cluster_index_result[key]]), axis=0),
        }
        centroid = discard_set[key][2] / discard_set[key][1]
        var = discard_set[key][3] / discard_set[key][1] - np.square(centroid)
        std_dev = np.sqrt(var)
        # Standard deviation
        discard_set[key][4] = std_dev
        # Centroid
        discard_set[key][5] = centroid

    """
    The initialization of DS has finished
    You have K numbers of DS clusters (from Step 5) and some numbers of RS (from Step 3)
    """

    # STEP 6: Run K-Means on the points in the RS with a large K
    # (e.g., 5 times of the number of the input clusters) to generate CS (clusters with more than one points) and
    # RS (clusters with only one point)
    RS_model_feed = [retained_set[key] for key in retained_set.keys()]
    # It seems that n_clusters=n_cluster * 5 is too large
    # k_means = KMeans(n_clusters=n_cluster * 5, random_state=553)
    k_means = KMeans(n_clusters=math.ceil(len(np.array(RS_model_feed))/2), random_state=553)
    cluster_result = k_means.fit_predict(np.array(RS_model_feed))
    cluster_index_result = get_cluster_indices(cluster_result)

    # Generate compression set
    compression_set = dict()
    for key in cluster_index_result.keys():
        if len(cluster_index_result[key]) > 1:
            compression_set = {key: {}}
            get_compression_set(compression_set, key, cluster_index_result[key], np.array(RS_model_feed),
                                retained_set, RS_model_feed)
    # Remove the points in the compression set from the retained set
    for key, cluster in cluster_index_result.items():
        if len(cluster) > 1:
            for i in cluster:
                retained_set_keys = list(retained_set.keys())
                point_to_remove = retained_set_keys[list(retained_set.values()).index(RS_model_feed[i])]
                del retained_set[point_to_remove]
    # Update retained set
    RS_model_feed = [retained_set[key] for key in retained_set.keys()]

    # Write intermediate result to file
    with open(output_file, "w") as f:
        f.write("The intermediate results:\n")
        # Call the function to write the intermediate result
        write_intermediate_result(f, 1, discard_set, compression_set, retained_set)

    """
    REPEAT STEPS 7-12
    """
    FINAL_ROUND = 5
    for round_track in range(2, FINAL_ROUND + 1):
        # STEP 7: Load another 20% of the data randomly
        start_index = end_index
        end_index = data_size if round_track == FINAL_ROUND else start_index + TWENTY_PERCENT
        next_pass = data_raw[start_index:end_index]
        last_ctr = point_ctr

        # Prepare data for K-Means
        model_feed, point_ctr, point_index_map, cluster_index_map = preprocess_data(next_pass, point_ctr,
                                                                                    point_index_map, cluster_index_map)

        def assign_to_closest_cluster(point, cluster_set, cluster_index, update_cluster_stats):
            closest_distance = find_closest_cluster(point, cluster_set)
            # STEP 8: For the new points, compare them to each of the DS using the Mahalanobis Distance and assign them
            if closest_distance > -1:
                update_cluster_stats(cluster_set, cluster_index, point, closest_distance)
                return True
            return False

        for j, point in enumerate(model_feed):
            cluster_index = cluster_index_map[last_ctr + j]

            # STEP 9: For the new points that are not assigned to DS clusters, using the Mahalanobis Distance and
            # assign the points to the nearest CS clusters if the distance is < 2*sqrt(DIMENSION)
            if not assign_to_closest_cluster(point, discard_set, cluster_index, update_cluster_stats):

                # Step 10: For the new points that are not assigned to a DS cluster or a CS cluster, assign them to RS
                if not assign_to_closest_cluster(point, compression_set, cluster_index, update_cluster_stats):
                    retained_set[cluster_index] = list(point)
                    RS_model_feed.append(list(point))

        # Step 11: Run K-Means on the RS with a large K (e.g., 5 times of the number of the input clusters) to
        # generate CS (clusters with more than one points) and RS (clusters with only one point)
        # k_means = KMeans(n_clusters=n_cluster * 5, random_state=553)
        k_means = KMeans(n_clusters=math.ceil(len(np.array(RS_model_feed)) / 2), random_state=553)
        cluster_result = k_means.fit_predict(np.array(RS_model_feed))
        cluster_index_result = get_cluster_indices(cluster_result)

        compression_sets = []
        for cluster_id, cluster_indices in cluster_index_result.items():
            if len(cluster_indices) > 1:
                new_cluster_id = cluster_id
                while new_cluster_id in compression_set:
                    new_cluster_id += 1
                new_compression_set = {new_cluster_id: {}}
                get_compression_set(new_compression_set, new_cluster_id, cluster_indices, np.array(RS_model_feed),
                                    retained_set, RS_model_feed)
                compression_sets.append(new_compression_set)
        # compression_set = {}
        for cs in compression_sets:
            compression_set.update(cs)

        # Remove the points in the compression set from the retained set
        for key, cluster in cluster_index_result.items():
            if len(cluster) > 1:
                for i in cluster:
                    point_to_remove = point_index_map[str(RS_model_feed[i])]
                    if point_to_remove in retained_set.keys():
                        del retained_set[point_to_remove]
        # Update retained set
        RS_model_feed = [retained_set[key] for key in retained_set.keys()]

        # Step 12. Merge CS clusters that have a Mahalanobis Distance < 2*sqrt(DIMENSION)
        Closest_CS = get_nearest_cluster(compression_set, compression_set)

        for cs_key in Closest_CS.keys():
            if cs_key != Closest_CS[cs_key] \
                    and Closest_CS[cs_key] in compression_set.keys() \
                    and cs_key in compression_set.keys():
                # Merge the two clusters
                merge_cluster(compression_set, cs_key, Closest_CS[cs_key])

        # If this is the last run (after the last chunk of data), merge CS clusters with DS clusters that have a
        # Mahalanobis Distance < 2*sqrt(DIMENSION)
        if round_track == FINAL_ROUND:
            Closest_CS = get_nearest_cluster(compression_set, discard_set)
            for CS_key in Closest_CS.keys():
                if CS_key in compression_set.keys() and Closest_CS[CS_key] in discard_set.keys():
                    # Merge CS with DS
                    discard_set[Closest_CS[CS_key]] = {
                        0: discard_set[Closest_CS[CS_key]][0] + compression_set[CS_key][0],
                        1: discard_set[Closest_CS[CS_key]][1] + compression_set[CS_key][1],
                        2: [x + y for x, y in zip(discard_set[Closest_CS[CS_key]][2], compression_set[CS_key][2])],
                        3: [x + y for x, y in zip(discard_set[Closest_CS[CS_key]][3], compression_set[CS_key][3])],
                    }
                    discard_set[Closest_CS[CS_key]][4] = np.sqrt(
                        (np.array(discard_set[Closest_CS[CS_key]][3]) / discard_set[Closest_CS[CS_key]][1]) -
                        ((np.array(discard_set[Closest_CS[CS_key]][2]) / discard_set[Closest_CS[CS_key]][1]) ** 2))
                    discard_set[Closest_CS[CS_key]][5] = [x / discard_set[Closest_CS[CS_key]][1] for x in
                                                          discard_set[Closest_CS[CS_key]][2]]
                    # del compression_set[CS_key]
        # Write intermediate result to file
        with open(output_file, "a") as f:
            write_intermediate_result(f, round_track, discard_set, compression_set, retained_set)

    final_result = {}
    # Obtain the final clustering result
    for key, cluster in discard_set.items():
        for point in cluster[0]:
            final_result[point] = key
    for key, cluster in compression_set.items():
        for point in cluster[0]:
            final_result[point] = -1
    for key, cluster in retained_set.items():
        final_result[key] = -1

    # Sort the final result by key
    sorted_dict = sorted(final_result.items(), key=lambda x: int(x[0]))

    # Write final result to file
    with open(output_file, "a") as f:
        f.write("\n")
        f.write("The clustering results:\n")
        for key, value in sorted_dict:
            f.write("{},{}\n".format(key, value))

    # Run time
    # print("Duration: {}".format(time.time() - start_time))
