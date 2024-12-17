import json
import os
import sys
import time
import warnings

import numpy as np
import pyspark
import xgboost as xgb
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")

sc = pyspark.SparkContext(appName="Competition")

sc.setLogLevel("WARN")

folder_path = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

def preprocess_data(system_type):
    yelp_train = sc.textFile(os.path.join(folder_path, 'yelp_train.csv')).map(lambda x: x.split(","))
    yelp_val = sc.textFile(test_file).map(lambda x: x.split(","))
    header = yelp_train.first()
    yelp_train = yelp_train.filter(lambda x: x != header).map(lambda x: (x[0], x[1], float(x[2])))
    header = yelp_val.first()
    yelp_val = yelp_val.filter(lambda x: x != header)
    business = sc.textFile(os.path.join(folder_path, "business.json")).map(lambda x: json.loads(x))
    user = sc.textFile(os.path.join(folder_path, "user.json")).map(lambda x: json.loads(x))
    

    if system_type == 'item':
        data_grouped = yelp_train.map(lambda row: (row[0], [(row[1], row[2])])).reduceByKey(lambda a, b: a + b)   # (user_id, [(business_id, rating)])
        business_user = yelp_train.map(lambda row: (row[1], [(row[0], row[2])])).reduceByKey(lambda a, b: a + b)
        business_user_dict = business_user.collectAsMap()                                                   # {business1_id: [(user1_id, rating1), ...], ...}
        user_business_dict = data_grouped.collectAsMap()
        business_avg = yelp_train.map(lambda row: (row[1], (float(row[2]), 1))) \
                            .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                                .mapValues(lambda x: x[0] / x[1]).collectAsMap()                                                # [(business1_id, avg_rating1), ...]
        user_avg = yelp_train.map(lambda row: (row[0], (float(row[2]), 1))) \
                                .reduceByKey(lambda a, b: (a[0] + b[0], a[1] + b[1])) \
                                .mapValues(lambda x: x[0] / x[1]).collectAsMap()                                 # {user1_id: avg_rating1, ...}
        avg_over_all_businesses = sum(business_avg.values()) / len(business_avg)
        avg_over_all_users = sum(user_avg.values()) / len(user_avg)
        return yelp_val, business_user_dict, user_business_dict, \
            business_avg, user_avg, avg_over_all_businesses, avg_over_all_users

    
    elif system_type == 'model':
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
    elif system_type == 'hybrid':
        return yelp_val

def calculate_rmse(pred_file, truth_file):
    if type(pred_file) == str:
        pred_file = sc.textFile(pred_file).map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]))
        header = pred_file.first()
        pred_file = pred_file.filter(lambda row: row != header).sortByKey()
    truth_file = sc.textFile(os.path.join(folder_path, truth_file)).map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]) if len(row) == 3 else row)
    header = truth_file.first()
    truth_file = truth_file.filter(lambda row: row != header).sortByKey()
    matching = pred_file.join(truth_file)
    rmse = matching.map(lambda row: (float(row[1][0]) - float(row[1][1])) ** 2).mean() ** (1/2)
    return rmse


def item_based():

    yelp_val, business_user_dict, user_business_dict, \
        business_avg, user_avg, avg_over_all_businesses, avg_over_all_users = preprocess_data('item')

    def pearson_correlation(business_avg, business_user_dict, business1, business2, avg_over_all_businesses):
        business1_avg = business_avg.get(business1, avg_over_all_businesses)
        business2_avg = business_avg.get(business2, avg_over_all_businesses)
        business1_users = business_user_dict.get(business1, [])
        business2_users = business_user_dict.get(business2, [])
        common_users = set([user1 for user1, _ in business1_users]).intersection(set([user2 for user2, _ in business2_users]))
        if len(common_users) <= 2:
            return 'x'
        business1_ratings = np.array([rating1 for user1, rating1 in business1_users if user1 in common_users])
        business2_ratings = np.array([rating2 for user2, rating2 in business2_users if user2 in common_users])
        business1_ratings -= business1_avg
        business2_ratings -= business2_avg
        numerator = np.dot(business1_ratings, business2_ratings)
        denominator = np.linalg.norm(business1_ratings) * np.linalg.norm(business2_ratings)
        if denominator == 0:
            return 'x'
        p = 1.5
        return (numerator / denominator) * (abs(numerator / denominator) ** (p - 1))

    def predict_rating(user, business, user_business_dict, business_user_dict, business_avg, \
                       user_avg, avg_over_all_businesses, avg_over_all_users):
        user_businesses = user_business_dict.get(user, [])
        if not user_businesses:
            return np.mean([user_avg.get(user, avg_over_all_users), business_avg.get(business, avg_over_all_businesses)])
        numerator = 0
        denominator = 0
        for business1, rating1 in user_businesses:
            if business1 == business:
                continue
            similarity = pearson_correlation(business_avg, business_user_dict, business, business1, avg_over_all_businesses)
            if type(similarity) == str:
                return np.mean([user_avg.get(user, avg_over_all_users), business_avg.get(business, avg_over_all_businesses)])
            numerator += similarity * (rating1 - business_avg.get(business1, avg_over_all_businesses))
            denominator += abs(similarity)
        if denominator == 0:
            return np.mean([user_avg.get(user, avg_over_all_users), business_avg.get(business, avg_over_all_businesses)])
        return np.mean([user_avg.get(user, avg_over_all_users), business_avg.get(business, avg_over_all_businesses)]) + (numerator / denominator)

    def write_preds():
        preds = yelp_val.map(lambda row: (row[0], row[1], predict_rating(row[0], row[1], \
                                                                         user_business_dict, business_user_dict, \
                                                                            business_avg, user_avg, \
                                                                                avg_over_all_businesses, avg_over_all_users))).collect()
        # normalize predictions
        preds = [(user, business, max(0, min(5, prediction))) for user, business, prediction in preds]
        with open('item_output.csv', 'w') as f:
            f.write('user_id, business_id, prediction\n')
            for user, business, prediction in preds:
                f.write(f'{user},{business},{prediction}\n')
    write_preds()



def xgb_model_based():

    def write_preds(data_val_id, preds_val):
        preds = []
        for i in range(len(data_val_id)):
            preds.append((data_val_id[i][0][0], data_val_id[i][0][1], preds_val[i]))
        with open('model_output.csv', 'w') as f:
            f.write('user_id, business_id, prediction\n')
            for user, business, prediction in preds:
                f.write(f'{user},{business},{prediction}\n')
    print('preprocessing model data')
    data_train, data_val_id, data_val = preprocess_data('model')
    data_array = np.array(data_train.collect()).astype(float)
    print('normalizing input vars')
    rs = RobustScaler()

    X_train = np.array(data_array[:, 1:])
    X_train = rs.fit_transform(X_train)
    y_train = np.array(data_array[:, 0])

    X_val = np.array(data_val.collect()).astype(float)
    X_val = rs.transform(X_val)

    #param = {
    #    'max_depth': 0,
    #    'eta': 0.02,
    #    'min_child_weight': 500,
    #    'subsample': 0.7,
    #    'lambda': 1,
    #    'colsample_bytree': 0.5,
    #    'gamma': 1,
    #    'objective': 'reg:linear',
    #    'eval_metric': 'rmse',
    #    'n_estimators': 3000,
    #}

    print('fitting model')
    gbt = xgb.XGBRegressor()#**param)
    gbt.fit(X_train, y_train, verbose=0)
    print('predicting')
    preds_test = gbt.predict(X_val)
    print('writing to file')
    write_preds(data_val_id.collect(), preds_test)
    # rmse_real = calculate_rmse(output_file, test_file)
    # print('RMSE (Model-Based):', rmse_real)



def weighted_hybrid_recommender():
    item_output = sc.textFile('item_output.csv').map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]))
    header = item_output.first()
    item_output = item_output.filter(lambda row: row != header).sortByKey()
    model_output = sc.textFile('model_output.csv').map(lambda row: row.split(',')).map(lambda row: ((row[0], row[1]), row[2]))
    header = model_output.first()
    model_output = model_output.filter(lambda row: row != header).sortByKey()
    matching = item_output.join(model_output)
    hybrid_preds = matching.map(lambda row: (row[0][0], row[0][1], (0.05 * float(row[1][0])) + ((1 - 0.05) * float(row[1][1])))).collect()
    with open(output_file, 'w') as f:
        f.write('user_id, business_id, prediction\n')
        for user, business, prediction in hybrid_preds:
            f.write(f'{user},{business},{prediction}\n')
    os.remove('item_output.csv')
    os.remove('model_output.csv')    



if __name__ == '__main__':
    start = time.time()
    item_start = time.time()
    print('\n\n\nItem-Based Recommender System:\n')
    item_based()
    item_end = time.time()
    print('Item-Based Duration:', round(item_end - item_start), 'seconds')
    model_start = time.time()
    print('\n\n\nXGBoost Model-Based Recommender System:\n')
    xgb_model_based()
    model_end = time.time()
    print('XGBoost Model-Based Duration:', round(model_end - model_start), 'seconds')
    print('\n\n\nHybrid Recommender System:\n')
    hybrid_start = time.time()
    weighted_hybrid_recommender()
    hybrid_end = time.time()
    print('Hybrid Duration:', round(hybrid_end - hybrid_start), 'seconds')
    end = time.time()
    print('\n\n\nTotal Duration:', round(end - start), 'seconds\n\n\n')
    sc.stop()