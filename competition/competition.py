import json
import numpy as np
import os
import pandas as pd
import pickle
import random
import sys
import time
import xgboost as xgb

from pyspark import SparkContext
# from sklearn import metrics
# from sklearn.model_selection import GridSearchCV

# Spark configuration
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
sc = SparkContext('local[*]', 'competition')
sc.setLogLevel("ERROR")

# Set random seed
random.seed(42)


"""
Method Description:
The model-based recommendation system is implemented by using the XGBoost algorithm for this competition.
To improve the model result from HW3, I include 11 more features in the model, as total 27 features are used.
The parameters of the XGBoost model are tuned by using the GridSearchCV function. The tuning process takes two steps.
First, I tune the parameters in larger ranges using 5-fold cross validation with fewer combinations.
Then, I tune the parameters in smaller ranges based on the results from the first step and 5-fold cross validation.
The model is then saved as a pickle file, which can be found in the same directory as this file.

Error Distribution:
>=0 and <1:  102162
>=1 and <2:  32924
>=2 and <3:  6140
>=3 and <4:  818
>=4:  0

RMSE：
0.9795075385454441

Execution Time:
113.56228876113892
"""


def preprocess_csv(input_rdd):
    output = input_rdd.map(lambda x: x.split(',')).collect()
    output_df = pd.DataFrame(output)

    # Use the first row as the header
    header = output_df.iloc[0]
    output_df = output_df[1:]
    output_df.columns = header
    return output_df


def price_range(attributes, price_range_key):
    if attributes is None:
        return random.randint(1, 4)
    elif price_range_key in attributes.keys():
        return int(attributes.get(price_range_key))
    else:
        return random.randint(1, 4)


def accept_credit_card(attributes, credit_card_key):
    if attributes is None:
        return 0
    elif credit_card_key in attributes.keys():
        credit_card_status = attributes.get(credit_card_key)
        if credit_card_status == 'False':
            return 0
        else:
            return 1
    else:
        return random.randint(0, 1)


def by_appointment_only(attributes, appointments_key):
    if attributes is None:
        return 0
    elif appointments_key in attributes.keys():
        appointments_status = attributes.get(appointments_key)
        if appointments_status == 'False':
            return 0
        else:
            return 1
    else:
        return random.randint(0, 1)


def restaurants_reservations(attributes, reservation_key):
    if attributes is None:
        return 0
    elif reservation_key in attributes.keys():
        reservation_status = attributes.get(reservation_key)
        if reservation_status == 'False':
            return 0
        else:
            return 1
    else:
        return random.randint(0, 1)


def restaurants_table_service(attributes, table_service_key):
    if attributes is None:
        return 0
    elif table_service_key in attributes.keys():
        table_service_status = attributes.get(table_service_key)
        if table_service_status == 'False':
            return 0
        else:
            return 1
    else:
        return random.randint(0, 1)


def restaurants_wheelchair(attributes, wheelchair_key):
    if attributes is None:
        return 0
    elif wheelchair_key in attributes.keys():
        wheelchair_status = attributes.get(wheelchair_key)
        if wheelchair_status == 'False':
            return 0
        else:
            return 1
    else:
        return random.randint(0, 1)


def user_elite_status(elite_status):
    if elite_status == 'None':
        return 0
    else:
        return len(elite_status.split(','))


def user_friends_count(friends):
    if friends == 'None':
        return 0
    else:
        return len(friends.split(','))


def preprocess_business(input_rdd):
    output_rdd = input_rdd.map(lambda x: ((x['business_id']),
                                          (x['stars'], x['review_count'], x['latitude'], x['longitude'],
                                           price_range(x['attributes'], 'RestaurantsPriceRange2'),
                                           accept_credit_card(x['attributes'], 'BusinessAcceptsCreditCards'),
                                           by_appointment_only(x['attributes'], 'ByAppointmentOnly'),
                                           restaurants_reservations(x['attributes'], 'RestaurantsReservations'),
                                           restaurants_table_service(x['attributes'], 'RestaurantsTableService'),
                                           restaurants_wheelchair(x['attributes'], 'WheelchairAccessible')
                                           ))
                               )
    return output_rdd


def preprocess_user(input_rdd):
    output_rdd = input_rdd.map(lambda x: ((x['user_id']),
                                          (x['review_count'], user_friends_count(x['friends']),
                                           x['useful'], x['funny'], x['cool'], x['fans'],
                                           user_elite_status(x['elite']), x['average_stars'],
                                           x['compliment_hot'], x['compliment_profile'], x['compliment_list'],
                                           x['compliment_note'], x['compliment_plain'], x['compliment_cool'],
                                           x['compliment_funny'], x['compliment_writer'], x['compliment_photos']))
                               )
    return output_rdd


def feature_construction(train_raw, business_raw, user_raw):
    # In case we don't have records for some business_id or user_id
    # Use average/random values to fill in based on attributes

    # Create empty lists to store features
    work_dict = {elem: list() for elem in attribute_names}

    # Use business_id and user_id as the key to construct features
    for business_id in train_raw['business_id']:
        if business_id in business_raw.keys():
            for index in range(len(attribute_names[:10])):
                work_dict[attribute_names[index]].append(business_raw.get(business_id)[index])
        else:
            pass
            # Fill in missing values with average/random values
            # print("Missing business_id: " + business_id)

    for user_id in train_raw['user_id']:
        if user_id in user_raw.keys():
            for index in range(len(attribute_names[10:])):
                work_dict[attribute_names[index + 10]].append(user_raw.get(user_id)[index])
        else:
            pass
            # Fill in missing values with average/random values
            # print("Missing user_id: " + user_id)

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
    # folder_path = "../resource/asnlib/publicdata/"
    # test_file_name = "../resource/asnlib/publicdata/yelp_val.csv"
    # output_file_name = "../output/competition_result.csv"

    # Files to be used for model training: yelp_train.csv, business.json, user.json
    train_file_path = folder_path + "yelp_train.csv"
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
    attribute_names = ['business_stars', 'business_review_count', 'latitude', 'longitude',
                       'business_price_range', 'business_credit_card', 'business_appointment_only',
                       'business_reservations', 'business_table_service', 'business_wheelchair',
                       'user_review_count', 'user_friends', 'user_useful', 'user_funny', 'user_cool', 'user_fans',
                       'user_elite', 'user_average_stars', 'user_compliment_hot', 'user_compliment_profile',
                       'user_compliment_list', 'user_compliment_note', 'user_compliment_plain', 'user_compliment_cool',
                       'user_compliment_funny', 'user_compliment_writer', 'user_compliment_photos']
    print(len(attribute_names))

    # Apply feature construction and format dataframe
    train_feature = feature_construction(train_df, business_data, user_data)
    train_feature = format_df(train_feature)
    test_feature = feature_construction(test_df, business_data, user_data)
    test_feature = format_df(test_feature)

    X_train = train_feature.drop(['stars'], axis=1)
    y_train = train_feature['stars']
    X_test = test_feature.drop(['stars'], axis=1)
    y_test = test_feature['stars']

    # Set parameters for GridSearchCV
    # parameters = {'max_depth': [3, 5, 7],
    #               'learning_rate': [0.05, 0.1, 0.2],
    #               'n_estimators': [500, 1000, 2000]}
    # xgboost_model = xgb.XGBRegressor(verbosity=0, seed=42)
    # clf = GridSearchCV(xgboost_model, parameters, scoring='neg_mean_squared_error', cv=5, verbose=3, n_jobs=-1)
    # clf.fit(X_train, y_train)
    #
    # print("Best parameters:", clf.best_params_)
    # # Best parameters: {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 2000}
    # # Best parameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000}
    # # Best parameters: {'learning_rate': 0.05, 'max_depth': 5, 'n_estimators': 1000}
    # print("Lowest RMSE: ", (-clf.best_score_) ** (1 / 2.0))
    # # Lowest RMSE:  0.984390795778219
    # # Lowest RMSE:  0.9832238282458702
    # # Lowest RMSE:  0.9825453397427384

    # Train model
    xgboost_model = xgb.XGBRegressor(learning_rate=0.05, max_depth=5, n_estimators=1000)
    xgboost_model.fit(X_train, y_train)

    # Save model
    # pickle.dump(xgboost_model, open("xgboost_model.pickle.dat", "wb"))

    # Load model
    # xgboost_model = pickle.load(open("xgboost_model.pickle.dat", "rb"))

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

    # Error distribution - Thanks Cameron!
    y_star = y_test.values.tolist()
    diff_0_1 = 0
    diff_1_2 = 0
    diff_2_3 = 0
    diff_3_4 = 0
    diff_gr_4 = 0

    for i in range(len(y_star)):
        temp = abs(y_star[i] - predictions[i])
        if temp < 1:
            diff_0_1 += 1
        elif temp < 2:
            diff_1_2 += 1
        elif temp < 3:
            diff_2_3 += 1
        elif temp < 4:
            diff_3_4 += 1
        else:
            diff_gr_4 += 1

    print(">=0 and <1: ", diff_0_1)
    print(">=1 and <2: ", diff_1_2)
    print(">=2 and <3: ", diff_2_3)
    print(">=3 and <4: ", diff_3_4)
    print(">=4: ", diff_gr_4)

    # Calculate RMSE
    # rmse = np.sqrt(metrics.mean_squared_error(y_test, predictions))
    # print("RMSE: {}".format(rmse))

    # Running time
    print("Duration: {}".format(time.time() - start_time))
