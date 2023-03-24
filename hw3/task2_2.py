import json
import numpy as np
import os
import pandas as pd
import random
import sys
import time
import xgboost as xgb

from pyspark import SparkContext
# from sklearn import metrics, preprocessing

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'task2_2')
sc.setLogLevel("WARN")


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
    # test_file_name = "../asnlib/publicdata/yelp_val.csv"
    # output_file_name = "task2_2_output.csv"

    # Files to be used for model training: yelp_train.csv, business.json, user.json
    train_file_path = os.path.join(folder_path, "yelp_train.csv")
    business_file_path = os.path.join(folder_path, "business.json")
    user_file_path = os.path.join(folder_path, "user.json")

    # Read train data
    train_data_rdd = sc.textFile(train_file_path)
    train_df = preprocess_csv(train_data_rdd)
    # print(train_df.head())

    # Read and preprocess test data
    test_data_rdd = sc.textFile(test_file_name)
    test_df = preprocess_csv(test_data_rdd)
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
    xgboost_model = xgb.XGBRegressor(max_depth=10, learning_rate=0.1, n_estimators=100)
    xgboost_model.fit(X_train, y_train)

    # Predict on test data
    predictions = xgboost_model.predict(X_test)

    result = pd.DataFrame({
        "user_id": user_id_record,
        "business_id": business_id_record,
        "prediction": predictions
    })

    # Save output
    result.to_csv(output_file_name, header=['user_id', ' business_id', ' prediction'], index=False)

    # For local testing
    # RMSE and running time
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    # print("RMSE: {}".format(rmse))
    # print("Duration: {}".format(time.time() - start_time))
