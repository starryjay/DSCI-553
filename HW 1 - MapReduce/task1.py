import json
import pyspark

sc = pyspark.SparkContext()
test_review_rdd = sc.textFile('./resource/lib/publicdata/review.json')

# Question A
n_review = test_review_rdd.count()

# Question B
n_review_2018 = test_review_rdd.map(lambda l: json.loads(l)).filter(lambda r: r['date'].startswith('2018')).count()

# Question C
n_user = test_review_rdd.map(lambda u: json.loads(u)['user_id']).distinct().count()

# Question D
top10_user = test_review_rdd.map(lambda u: json.loads(u)).map(lambda u: (u['user_id'], 1)).reduceByKey(lambda r1, r2: r1 + r2).sortBy(lambda r: r[1], ascending=False).map(lambda t: [t[0], t[1]]).take(10)

# Question E
n_business = test_review_rdd.map(lambda b: json.loads(b)['business_id']).distinct().count()

# Question F
top10_reviewed = test_review_rdd.map(lambda b: json.loads(b)).map(lambda b: (b['business_id'], 1)).reduceByKey(lambda r1, r2: r1 + r2).sortBy(lambda r: r[1], ascending=False).map(lambda t: [t[0], t[1]])
top10_business = top10_reviewed.take(10)

with open('task1.json', 'w') as f:
    json.dump({'n_review': n_review, 'n_review_2018': n_review_2018, 'n_user': n_user, 'top10_user': top10_user, 'n_business': n_business, 'top10_business': top10_business}, f)