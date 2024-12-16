import json
import os
import sys
import time
import warnings
import random

import numpy as np
import pyspark
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

sc = pyspark.SparkContext(appName="Competition")

sc.setLogLevel("WARN")

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

def preprocess_data():
    yelp_train = sc.textFile(os.path.join(folder_path, "yelp_train.csv")).map(lambda x: x.split(","))
    yelp_val = sc.textFile(os.path.join(folder_path, test_file)).map(lambda x: x.split(","))
    header = yelp_train.first()
    yelp_train = yelp_train.filter(lambda x: x != header).map(lambda x: (x[0], x[1], float(x[2])))
    header = yelp_val.first()
    yelp_val = yelp_val.filter(lambda x: x != header).map(lambda x: (x[0], x[1], float(x[2])) if len(x) == 3 else x)
    business = sc.textFile(os.path.join(folder_path, "business.json")).map(lambda x: json.loads(x))
    user = sc.textFile(os.path.join(folder_path, "user.json")).map(lambda x: json.loads(x))
    checkin = sc.textFile(os.path.join(folder_path, "checkin.json")).map(lambda x: json.loads(x))        
    def get_day_of_week(day_date):
        return day_date.split('-')[0]
    
    def weekday_order(day_tuple=None, day=None):
        weekdays = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
        if day_tuple:
            return weekdays[day_tuple[0]]
        return weekdays[day]
    
    most_freq_days = checkin.flatMap(lambda x: [((x['business_id'], get_day_of_week(day_date)), count) for day_date, count in x['time'].items()])\
        .reduceByKey(lambda a, b: a + b).map(lambda x: (x[0][0], [(x[0][1], x[1])])).reduceByKey(lambda a, b: a + b)\
            .map(lambda row: (row[0], sorted(row[1], key=weekday_order))).map(lambda row: (row[0], max(row[1], key=lambda x: x[1])[0]))\
            .map(lambda x: (x[0], weekday_order(day=x[1])))
    average_stars_user = user.map(lambda x: (x['user_id'], float(x['average_stars'])))
    business_info = business.map(lambda x: (x['business_id'], (float(x['stars']), int(x['review_count']), x['is_open'])))
    all_attributes = business.flatMap(lambda x: x['attributes'].items() if x['attributes'] else [])\
        .flatMap(lambda x: x[1].items() if type(x[1]) == dict else [(x[0], x[1])]).filter(lambda x: '{' not in x[1] and 'False' in x[1])\
            .distinct().map(lambda x: (x[0], [x[1]])).reduceByKey(lambda a, b: a + b)\
            .map(lambda x: (x[0], list(set(x[1])))).sortByKey().collectAsMap()
    business_attributes = business.map(lambda x: (x['business_id'], list(x['attributes'].items()) if x['attributes'] else []))\
        .map(lambda x: (x[0], [(key, value) for key, value in x[1] if (value == "True" or value == "False")] if x[1] else []))\
            .filter(lambda x: x[1] != []).map(lambda x: (x[0], [(key, 1) if value == "True" else (key, 0) for key, value in x[1]]))\
            .map(lambda x: (x[0], x[1], [(key, -1) for key in all_attributes.keys() if key not in [k for k, _ in x[1]]]))\
                .map(lambda x: (x[0], x[1] + x[2])).map(lambda x: (x[0], sorted(x[1], key=lambda x: x[0])))\
                    .map(lambda x: (x[0], ([val for _, val in x[1]]))) 
    business_data = business_info.join(most_freq_days).map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][1])))\
        .join(business_attributes).map(lambda x: (x[0], (x[1][0][0], x[1][0][1], x[1][0][2], x[1][0][3], *[i for i in x[1][1]])))
    user_data = yelp_train.map(lambda x: (x[0], (x[1], x[2]))).leftOuterJoin(average_stars_user).map(lambda x: (x[1][0][0], (x[0], x[1][0][1], x[1][1])))
    data = business_data.rightOuterJoin(user_data).map(lambda x: (x[0], *x[1][1], *x[1][0]) if None not in x[1] else (x[0], *x[1][1], x[1][0])).map(lambda x: ((x[1], x[0]), x[2:])) 
    user_data_val = yelp_val.map(lambda x: (x[0], x[1])).leftOuterJoin(average_stars_user).map(lambda x: (x[1][0], (x[0], x[1][1])))
    data_val_id = business_data.rightOuterJoin(user_data_val).map(lambda x: (x[0], *x[1][1], x[1][0])).map(lambda x: ((x[1], x[0]), x[2:]))       
    # (business_id, user_id, avg_user_stars, avg_business_stars, review_count, is_open, most_freq_day, \

    # 'AcceptsInsurance', 'BYOB', 'BikeParking', 'BusinessAcceptsBitcoin', 'BusinessAcceptsCreditCards', \
    # 'ByAppointmentOnly', 'Caters', 'CoatCheck', 'Corkage', 'DogsAllowed', 'DriveThru', 'GoodForDancing', \
    # 'GoodForKids', 'HappyHour', 'HasTV', 'Open24Hours', 'OutdoorSeating', 'RestaurantsCounterService', \
    # 'RestaurantsDelivery', 'RestaurantsGoodForGroups', 'RestaurantsReservations', 'RestaurantsTableService', \
    # 'RestaurantsTakeOut', 'WheelchairAccessible')
    mean_business_stars = round(data_val_id.filter(lambda x: x[1][1] is not None).map(lambda x: x[1][1][0]).mean(), 1)
    mean_review_count = round(data_val_id.filter(lambda x: x[1][1] is not None).map(lambda x: x[1][1][1]).mean())
    mode_is_open = data_val_id.filter(lambda x: x[1][1] is not None).map(lambda x: x[1][1][2]).map(lambda x: (x, 1))\
        .reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False).first()[0]
    mode_most_freq_day = data_val_id.filter(lambda x: x[1][1] is not None).map(lambda x: x[1][1][3]).map(lambda x: (x, 1))\
        .reduceByKey(lambda a, b: a + b).sortBy(lambda x: x[1], ascending=False).first()[0]
    data_val_id = data_val_id.map(lambda x: (x[0], x[1][0], (mean_business_stars if x[1][1] is None else x[1][1][0], \
                                                             mean_review_count if x[1][1] is None else x[1][1][1], \
                                                                mode_is_open if x[1][1] is None else x[1][1][2], \
                                                                mode_most_freq_day if x[1][1] is None else x[1][1][3], \
                                                                    tuple([-1 for _ in range(24)]) if x[1][1] is None else x[1][1][4:])))
    data_train = data.map(lambda x: (x[0], x[1][0], x[1][1], (mean_business_stars if x[1][2] is None else x[1][2], \
            mean_review_count if x[1][2] is None else x[1][3], mode_is_open if x[1][2] is None else x[1][4], \
            mode_most_freq_day if x[1][2] is None else x[1][5], tuple([-1 for _ in range(24)]) if x[1][2] is None else x[1][6:])))
    data_val = data_val_id.map(lambda x: (x[1], *x[2])).map(lambda x: (*x[0:5], *[int(i) for i in x[5]]))
    data_val_id = data_val_id.map(lambda x: (x[0], x[1], *x[2])).map(lambda x: (x[0], x[1:6], [int(i) for i in x[6]]))    
    data_train = data_train.map(lambda x: (x[1], x[2], *x[3])).map(lambda x: (*x[0:6], *[int(i) for i in x[6]]))
    return data_train, data_val_id, data_val

# def calculate_rmse(pred_file, truth_file):
#     if type(pred_file) == str:
#         pred_file = sc.textFile(pred_file).map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]))
#         header = pred_file.first()
#         pred_file = pred_file.filter(lambda row: row != header).sortByKey()
#     truth_file = sc.textFile(os.path.join(folder_path, truth_file)).map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]))
#     header = truth_file.first()
#     truth_file = truth_file.filter(lambda row: row != header).sortByKey()
#     matching = pred_file.join(truth_file)
#     rmse = matching.map(lambda row: (float(row[1][0]) - float(row[1][1])) ** 2).mean() ** (1/2)
#     return rmse

def xgb_model_based():

    def write_preds(data_val_id, preds_val):
        preds = []
        for i in range(len(data_val_id)):
            preds.append((data_val_id[i][0][0], data_val_id[i][0][1], preds_val[i]))
        with open(output_file, 'w') as f:
            f.write('user_id, business_id, prediction\n')
            for user, business, prediction in preds:
                f.write(f'{user},{business},{prediction}\n')
    print('preprocessing model data')
    data_train, data_val_id, data_val = preprocess_data()
    data_array = np.array(data_train.collect()).astype(float)
    print('normalizing input vars')
    rs = RobustScaler()

    X = np.array(data_array[:, 1:])
    y = np.array(data_array[:, 0])

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = rs.fit_transform(X_train)

    X_val = rs.transform(X_val)

    X_test = np.array(data_val.collect()).astype(float)
    X_test = rs.transform(X_test)

    param = {
        'max_depth': 0,
        'eta': 0.02,
        'min_child_weight': 500,
        'subsample': 0.7,
        'lambda': 1,
        'colsample_bytree': 0.5,
        'gamma': 1,
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'n_estimators': 3000,
        'early_stopping_rounds': 100
    }

    print('fitting model')
    gbt = xgb.XGBRegressor(**param)
    gbt.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=10)
    print('predicting')
    preds_test = gbt.predict(X_test)
    print('writing to file')
    write_preds(data_val_id.collect(), preds_test)
    # rmse_real = calculate_rmse(output_file, test_file)
    # print('RMSE (Model-Based):', rmse_real)

if __name__ == '__main__':
    model_start = time.time()
    print('\n\n\nXGBoost Model-Based Recommender System:\n')
    xgb_model_based()
    model_end = time.time()
    print('XGBoost Model-Based Duration:', round(model_end - model_start), 'seconds')
    sc.stop()