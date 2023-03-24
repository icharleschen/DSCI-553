import os
import sys
import time

from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task2_1')

# Define coefficients globally
DEFAULT_RATING = 2.5
NEIGHBOUR_LIMIT = 75
MAX_RATING = 5.0
MIN_RATING = 0.0


def generate_rating_dict(input_rdd, key_index, sub_key_index, rating_index):
    output_dict = input_rdd.map(lambda x: (x[key_index], (x[sub_key_index], float(x[rating_index]))))\
        .groupByKey()\
        .mapValues(dict)\
        .collectAsMap()
    return output_dict


def generate_average_rating_dict(input_rdd, key_index, rating_index):
    output_dict = input_rdd.map(lambda x: (x[key_index], float(x[rating_index])))\
        .groupByKey()\
        .mapValues(lambda x: sum(x) / len(x))\
        .collectAsMap()
    return output_dict


def PearsonCorrelation(neighbour_id, users_id, business_rating, avg_business_rating, business_dict, avg_business_dict):
    # Use try-except to handle missing keys and obtain two types of ratings
    try:
        business_neighbour_rating = business_dict.get(neighbour_id)
        avg_business_neighbour_rating = avg_business_dict.get(neighbour_id)
    except KeyError:
        return -1

    # Obtain the ratings of the current business and neighbour for the users in the neighbourhood and store result
    business_rating_record = list()
    neighbour_rating_record = list()
    for working_user_id in users_id:
        neighbour_ratings = business_neighbour_rating.get(working_user_id)
        if neighbour_ratings is not None:
            business_ratings = business_rating.get(working_user_id)
            if business_ratings is not None:
                business_rating_record.append(business_ratings)
                neighbour_rating_record.append(neighbour_ratings)

    if not business_rating_record:
        return float(avg_business_rating / avg_business_neighbour_rating)

    numerator, denominator_business, denominator_neighbour = 0, 0, 0
    # Plug-in the formula for Pearson Correlation
    for business_score, neighbour_score in zip(business_rating_record, neighbour_rating_record):
        temp_b_rating = business_score - avg_business_rating
        temp_n_rating = neighbour_score - avg_business_neighbour_rating
        numerator += temp_b_rating * temp_n_rating
        denominator_business += temp_b_rating ** 2
        denominator_neighbour += temp_n_rating ** 2

    denominator = (denominator_business * denominator_neighbour) ** 0.5

    # Handle special cases
    if denominator == 0:
        if numerator == 0:
            # Same direction
            pearson_correlation = 1.0
        else:
            pearson_correlation = -1.0
    else:
        pearson_correlation = numerator / denominator

    return pearson_correlation


def ItemBasedCF(input_rdd):
    # user_id, business_id, _ = input_rdd
    user_id, business_id = input_rdd

    # Handle cold start cases
    if business_id not in business_user_rating:
        if user_id not in user_business_rating:
            # Use average rating on scale of 0-5 when no information is available
            return user_id, business_id, DEFAULT_RATING
        # Use average rating of user when no information is available for business
        return user_id, business_id, user_average_rating.get(user_id)

    # Existing business
    users = list(business_user_rating.get(business_id, {}))
    business_rating = business_user_rating.get(business_id, {})
    avg_business_rating = business_average_rating.get(business_id, DEFAULT_RATING)

    if not user_business_rating.get(user_id):
        # Handle cold start case
        return user_id, business_id, avg_business_rating

    # Use existing users to get business ratings
    business_rating_list = list(user_business_rating.get(user_id))

    # Calculate Pearson Correlation
    if business_rating_list:
        coefficient_rating = list()
        for working_business_id in business_rating_list:
            working_rating = business_user_rating.get(working_business_id, {}).get(user_id)
            person_coefficient_value = PearsonCorrelation(working_business_id,
                                                          user_id,
                                                          business_rating,
                                                          avg_business_rating,
                                                          business_user_rating,
                                                          business_average_rating
                                                          )
            if 0 < person_coefficient_value < 1:
                coefficient_rating.append((person_coefficient_value, working_rating))
            elif person_coefficient_value >= 1:
                coefficient_rating.append((1 / person_coefficient_value, working_rating))

        coefficient_rating.sort(reverse=True)

        if not coefficient_rating:
            prediction = (user_average_rating.get(user_id, DEFAULT_RATING) + avg_business_rating) / 2
        else:
            neighbourhood = min(len(coefficient_rating), NEIGHBOUR_LIMIT)
            prediction_weight_sum = sum(
                coefficient * rating for coefficient, rating in coefficient_rating[:neighbourhood])
            pearson_coefficient_sum = sum(abs(coefficient) for coefficient, _ in coefficient_rating[:neighbourhood])
            prediction = prediction_weight_sum / pearson_coefficient_sum
        return user_id, business_id, min(MAX_RATING, max(MIN_RATING, prediction))

    # Cold start case when no information is available from user regarding business
    return user_id, business_id, avg_business_rating


if __name__ == "__main__":
    start_time = time.time()
    # Handle input arguments
    train_file_name = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # For local testing
    # train_file_name = "../asnlib/publicdata/yelp_train.csv"
    # test_file_name = "../asnlib/publicdata/yelp_val.csv"
    # output_file_name = "task2_1_output.csv"

    # Read train data
    train_data = sc.textFile(train_file_name)  # Consider add number of partitions to avoid crashing
    header = train_data.first()  # Remove header
    train_rdd = train_data.filter(lambda x: x != header) \
        .map(lambda x: x.split(','))

    # Create user pointing to business and rating dictionary
    # {user_id: {business_id, rating}}
    user_business_rating = generate_rating_dict(train_rdd, 0, 1, 2)

    # Create business pointing to user and rating dictionary
    # {business_id: {user_id, rating}}
    business_user_rating = generate_rating_dict(train_rdd, 1, 0, 2)

    # Create user average rating dictionary
    # {user_id: average_rating}
    user_average_rating = generate_average_rating_dict(train_rdd, 0, 2)

    # Create business average rating dictionary
    # {business_id: average_rating}
    business_average_rating = generate_average_rating_dict(train_rdd, 1, 2)

    # Handle test data
    test_data = sc.textFile(test_file_name)
    header_test = test_data.first()  # Remove header
    test_rdd = test_data.filter(lambda x: x != header_test)\
        .map(lambda x: x.split(','))

    prediction_rdd = test_rdd.map(ItemBasedCF)

    # Save output
    with open(output_file_name, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for line in prediction_rdd.collect():
            f.write("{},{},{}\n".format(str(line[0]), str(line[1]), str(line[2])))

    # For local testing
    # Compare result with sample output
    # result_rdd = sc.textFile(output_file_name)
    # header_result = result_rdd.first()
    # result_data = result_rdd.filter(lambda x: x != header_result)\
    #     .map(lambda x: x.split(','))
    #
    # output_data_formatted = result_data.map(lambda x: (((x[0]), (x[1])), float(x[2])))
    # test_data_formatted = test_rdd.map(lambda x: (((x[0]), (x[1])), float(x[2])))
    # diff_rdd = test_data_formatted.join(output_data_formatted).map(lambda x: (abs(x[1][0] - x[1][1])))
    #
    # diff_0_1 = diff_rdd.filter(lambda x: 0 <= x < 1).count()
    # diff_1_2 = diff_rdd.filter(lambda x: 1 <= x < 2).count()
    # diff_2_3 = diff_rdd.filter(lambda x: 2 <= x < 3).count()
    # diff_3_4 = diff_rdd.filter(lambda x: 3 <= x < 4).count()
    # diff_greater_than_4 = diff_rdd.filter(lambda x: x >= 4).count()
    # print(">=0 and <1: ", diff_0_1)
    # print(">=1 and <2: ", diff_1_2)
    # print(">=2 and <3: ", diff_2_3)
    # print(">=3 and <4: ", diff_3_4)
    # print(">=4: ", diff_greater_than_4)
    # RMSE_rdd = diff_rdd.map(lambda x: x ** 2).reduce(lambda x, y: x + y)
    # RMSE = (RMSE_rdd / output_data_formatted.count()) ** 0.5
    # print("RMSE: {}".format(RMSE))
    # print("Duration: {}".format(time.time() - start_time))
