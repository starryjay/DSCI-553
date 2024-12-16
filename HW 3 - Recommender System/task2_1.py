import sys
import warnings

import numpy as np
import pyspark

warnings.filterwarnings("ignore")

sc = pyspark.SparkContext(appName="Competition")

sc.setLogLevel("WARN")

train_file = sys.argv[1]
test_file = sys.argv[2]
output_file = sys.argv[3]

def preprocess_data():
    yelp_train = sc.textFile(train_file).map(lambda x: x.split(","))
    yelp_val = sc.textFile(test_file).map(lambda x: x.split(","))
    header = yelp_train.first()
    yelp_train = yelp_train.filter(lambda x: x != header).map(lambda x: (x[0], x[1], float(x[2])))
    header = yelp_val.first()
    yelp_val = yelp_val.filter(lambda x: x != header)
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

def item_based():

    yelp_val, business_user_dict, user_business_dict, \
        business_avg, user_avg, avg_over_all_businesses, avg_over_all_users = preprocess_data()

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
        with open(output_file, 'w') as f:
            f.write('user_id, business_id, prediction\n')
            for user, business, prediction in preds:
                f.write(f'{user},{business},{prediction}\n')
    write_preds()

if __name__ == '__main__':
    item_based()