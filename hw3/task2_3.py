import json
import os
import pandas as pd
import random
import sys
import time
import xgboost as xgb

from pyspark import SparkContext

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task2_3')
sc.setLogLevel("WARN")

"""
From Task2_1
"""
# Define coefficients globally
DEFAULT_RATING = 2.5
NEIGHBOUR_LIMIT = 75
MAX_RATING = 5.0
MIN_RATING = 0.0


def generate_rating_dict(input_rdd, key_index, sub_key_index, rating_index):
    output_dict = input_rdd.map(lambda x: (x[key_index], (x[sub_key_index], float(x[rating_index])))) \
        .groupByKey() \
        .mapValues(dict) \
        .collectAsMap()
    return output_dict


def generate_average_rating_dict(input_rdd, key_index, rating_index):
    output_dict = input_rdd.map(lambda x: (x[key_index], float(x[rating_index]))) \
        .groupByKey() \
        .mapValues(lambda x: sum(x) / len(x)) \
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


"""
From Task2_2
"""
def preprocess_csv(input_rdd):
    output = input_rdd.map(lambda x: x.split(',')).collect()
    output_df = pd.DataFrame(output)

    # Use the first row as the header
    header = output_df.iloc[0]
    output_df = output_df[1:]
    output_df.columns = header
    return output_df


def RestaurantsPriceRange(attributes, price_range_key):
    if attributes is None:
        return 0
    elif price_range_key in attributes.keys():
        return int(attributes.get(price_range_key))
    else:
        return 0


def preprocess_business(input_rdd):
    output_rdd = input_rdd.map(lambda x: ((x['business_id']),
                                          (x['stars'], x['review_count'],
                                           RestaurantsPriceRange(x['attributes'], 'RestaurantsPriceRange2')))
                               )
    return output_rdd


def preprocess_user(input_rdd):
    output_rdd = input_rdd.map(lambda x: ((x['user_id']),
                                          (x['review_count'], x['useful'], x['fans'], x['average_stars']))
                               )
    return output_rdd


def calculate_avg_rating(input_rdd, attribute_index):
    avg = input_rdd.map(lambda x: x[1][attribute_index]).mean()
    return avg


def feature_construction(train_raw, business_raw, user_raw):
    # In case we don't have record for some business_id or user_id
    # Use average/random values to fill in based on attributes

    # Create empty lists to store features
    work_dict = {elem: list() for elem in attribute_names}

    # Use business_id and user_id as the key to construct features
    for business_id in train_raw['business_id']:
        if business_id in business_raw.keys():
            for index in range(len(attribute_names[:3])):
                work_dict[attribute_names[index]].append(business_raw.get(business_id)[index])
        else:
            # Fill in missing values with average/random values
            work_dict[attribute_names[0]].append(avg_business_stars)
            work_dict[attribute_names[1]].append(avg_business_review_count)
            work_dict[attribute_names[2]].append(random.randint(0, 4))

    for user_id in train_raw['user_id']:
        if user_id in user_raw.keys():
            for index in range(len(attribute_names[3:])):
                work_dict[attribute_names[index + 3]].append(user_raw.get(user_id)[index])
        else:
            work_dict[attribute_names[3]].append(avg_user_review_count)
            work_dict[attribute_names[4]].append(avg_user_useful)
            work_dict[attribute_names[5]].append(avg_user_fans)
            work_dict[attribute_names[6]].append(avg_user_stars)

    # Add features to dataframe
    for index in range(len(attribute_names)):
        train_raw[attribute_names[index]] = work_dict[attribute_names[index]]

    return train_raw


def format_df(data_df):
    data_df = data_df.drop(['user_id', 'business_id'], axis=1)
    data_df = data_df.apply(pd.to_numeric)
    return data_df


if __name__ == "__main__":
    start_time = time.time()
    # Handle input arguments
    folder_path = sys.argv[1]
    test_file_name = sys.argv[2]
    output_file_name = sys.argv[3]

    # For local testing
    # folder_path = "../asnlib/publicdata/"
    # test_file_name = "../asnlib/publicdata/yelp_val_in.csv"
    # output_file_name = "task2_3_output.csv"

    # Files to be used for model training: yelp_train.csv, business.json, user.json
    train_file_path = folder_path + "yelp_train.csv"
    business_file_path = os.path.join(folder_path, "business.json")
    user_file_path = os.path.join(folder_path, "user.json")

    # Save temporary files for later combining result
    # item_based_result_path = "item_based_result.csv"
    # model_based_result_path = "model_based_result.csv"

    """
    Item Based Collaborative Filtering
    """
    # Read train data
    train_data = sc.textFile(train_file_path)  # Consider add number of partitions to avoid crashing
    header = train_data.first()  # Remove header
    train_rdd_IB = train_data.filter(lambda x: x != header) \
        .map(lambda x: x.split(','))

    # Create user pointing to business and rating dictionary
    # {user_id: {business_id, rating}}
    user_business_rating = generate_rating_dict(train_rdd_IB, 0, 1, 2)

    # Create business pointing to user and rating dictionary
    # {business_id: {user_id, rating}}
    business_user_rating = generate_rating_dict(train_rdd_IB, 1, 0, 2)

    # Create user average rating dictionary
    # {user_id: average_rating}
    user_average_rating = generate_average_rating_dict(train_rdd_IB, 0, 2)

    # Create business average rating dictionary
    # {business_id: average_rating}
    business_average_rating = generate_average_rating_dict(train_rdd_IB, 1, 2)

    # Handle test data
    test_data = sc.textFile(test_file_name)
    header_test = test_data.first()  # Remove header
    test_rdd_IB = test_data.filter(lambda x: x != header_test) \
        .map(lambda x: x.split(','))

    item_based_CF_prediction = test_rdd_IB.map(ItemBasedCF)\
        .map(lambda x: ((x[0], x[1]), float(x[2])))
    # print(item_based_CF_prediction.take(5))

    # print("Item Based CF Time: ", time.time() - start_time)
    new_time = time.time()

    """
    Model Based Collaborative Filtering
    """

    # Read train data
    train_df = preprocess_csv(train_data)
    # print(train_df.head())

    # Read and preprocess test data
    test_df = preprocess_csv(test_data)
    user_id_record = test_df['user_id'].tolist()
    business_id_record = test_df['business_id'].tolist()

    # Read and preprocess business data
    business_rdd = sc.textFile(business_file_path).map(json.loads)
    business_rdd = preprocess_business(business_rdd)
    business_data = business_rdd.collectAsMap()

    # Read and preprocess user data
    user_rdd = sc.textFile(user_file_path).map(json.loads)
    user_rdd = preprocess_user(user_rdd)
    user_data = user_rdd.collectAsMap()

    # Feature construction and datatype conversion
    attribute_names = ['business_stars', 'business_review_count', 'business_price_range',
                       'user_review_count', 'user_useful', 'user_fans', 'user_average_stars']

    # Average business data attributes
    avg_business_stars = calculate_avg_rating(business_rdd, 0)
    avg_business_review_count = calculate_avg_rating(business_rdd, 1)
    # Average user data attributes
    avg_user_review_count = calculate_avg_rating(user_rdd, 0)
    avg_user_useful = calculate_avg_rating(user_rdd, 1)
    avg_user_fans = calculate_avg_rating(user_rdd, 2)
    avg_user_stars = calculate_avg_rating(user_rdd, 3)

    # Apply feature construction and format dataframe
    train_feature = feature_construction(train_df, business_data, user_data)
    train_feature = format_df(train_feature)
    test_feature = feature_construction(test_df, business_data, user_data)
    test_feature = format_df(test_feature)

    X_train = train_feature.drop(['stars'], axis=1)
    y_train = train_feature['stars']
    X_test = test_feature.drop(['stars'], axis=1)
    # y_test = test_feature['stars']

    # Train model
    xgboost_model = xgb.XGBRegressor(max_depth=5, learning_rate=0.2, n_estimators=100)
    xgboost_model.fit(X_train, y_train)

    # Predict on test data
    model_based_prediction = xgboost_model.predict(X_test)

    # print("Model Based CF Time: ", time.time() - new_time)

    result = pd.DataFrame({
        "user_id": user_id_record,
        "business_id": business_id_record,
        "prediction": model_based_prediction
    })

    # Save temp output
    result.to_csv("temp_MB_result.csv", header=['user_id', ' business_id', ' prediction'], index=False)

    # Read result
    result_rdd = sc.textFile("temp_MB_result.csv")
    header_result = result_rdd.first()
    result_data = result_rdd.filter(lambda x: x != header_result)\
        .map(lambda x: x.split(','))\
        .map(lambda x: ((x[0], x[1]), float(x[2])))
    # print(result_data.take(5))

    # Hybrid recommendation system
    final_result = item_based_CF_prediction.join(result_data)\
        .map(lambda x: ((x[0]), float((x[1][0] * 0.1 + x[1][1] * 0.9))))
    # print(final_result.take(5))

    # Write output
    with open(output_file_name, 'w') as f:
        f.write("user_id, business_id, prediction\n")
        for line in final_result.collect():
            f.write(str(line[0][0]) + "," + str(line[0][1]) + "," + str(line[1]) + "\n")

    # For local testing
    # Compare result with sample output
    # result_rdd = sc.textFile(output_file_name)
    # header_result = result_rdd.first()
    # result_data = result_rdd.filter(lambda x: x != header_result)\
    #     .map(lambda x: x.split(','))
    #
    # output_data_formatted = result_data.map(lambda x: (((x[0]), (x[1])), float(x[2])))
    # test_data_formatted = test_rdd_IB.map(lambda x: (((x[0]), (x[1])), float(x[2])))
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
