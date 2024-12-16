import json
import pyspark
from task1 import test_review_rdd
import time

sc = pyspark.SparkContext()
business_rdd = sc.textFile('./resource/lib/publicdata/business.json')

# Question A
start_time_s = time.time()
avg_stars_spark_sort = (business_rdd
                        .join(test_review_rdd)
                        .map(lambda joined: (json.loads(joined[1][0])['city'], json.loads(joined[1][1])['stars']))
                        .groupByKey()
                        .map(lambda stars: sum(stars[1]) / len(stars[1]))
                        .sortBy(lambda averaged: averaged[1], ascending=False)
                        .map(lambda reviews: [reviews[0], reviews[1]])
                        .collect()
                        )
end_time_s = time.time()

# Question B
start_time_p = time.time()
avg_stars_unsorted = (business_rdd
                      .join(test_review_rdd)
                      .map(lambda joined: (json.loads(joined[1][0])['city'], json.loads(joined[1][1])['stars']))
                      .groupByKey()
                      .map(lambda stars: sum(stars[1]) / len(stars[1]))
                      .map(lambda reviews: [reviews[0], reviews[1]])
                      .collect()
                      )
average_stars_python_sort = sorted(avg_stars_unsorted, key=lambda stars: stars[1], reverse=True)
end_time_p = time.time()

with open('task3.txt', 'w') as f1:
    f1.write('city,stars\n')
    f1.write(str(avg_stars_spark_sort) + '\n')

with open('task3.json', 'w') as f2:
    json.dump({'m1': end_time_s - start_time_s, 'm2': end_time_p - start_time_p, 'reason': 'idk'}, f2)