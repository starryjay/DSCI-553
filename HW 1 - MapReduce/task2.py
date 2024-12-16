'''
Since processing large volumes of data requires performance optimizations, properly partitioning the
data for processing is imperative.
In this task, you will show the number of partitions for the RDD used for Task 1 Question F and the
number of items per partition.
Then you need to use a customized partition function to improve the performance of map and reduce
tasks. A time duration (for executing Task 1 Question F) comparison between the system default
partition and your customized partition (RDD built using the partition function) should also be shown in
your results.

Hint:
Certain operations within Spark trigger an event known as the shuffle. The shuffle is Spark's mechanism
for redistributing data so that it's grouped differently across partitions. This typically involves copying
data across executors and machines, making the shuffle a complex and costly operation. So, trying to
design a partition function to avoid the shuffle will improve the performance a lot.
'''

from task1 import test_review_rdd
from pyspark.sql.functions import spark_partition_id
import time
import json

start_time = time.time()
top10_reviewed = (test_review_rdd
                  .map(lambda b: json.loads(b))
                  .map(lambda b: (b['business_id'], 1))
                  .reduceByKey(lambda r1, r2: r1 + r2)
                  .sortBy(lambda r: r[1], ascending=False))
n_partition = top10_reviewed.getNumPartitions()
n_items = top10_reviewed.mapPartitions(lambda p: [sum(1 for i in p)]).collect()
end_time = time.time()

def custom_partition(business_id):
    return hash(business_id) % n_partition

start_time_customized = time.time()
top10_reviewed_customized = (test_review_rdd
                             .map(lambda b: json.loads(b))
                             .map(lambda b: (b['business_id'], 1))
                             .partitionBy(n_partition, custom_partition)
                             .reduceByKey(lambda r1, r2: r1 + r2)
                             .sortBy(lambda r: r[1], ascending=False))
n_partition_customized = top10_reviewed_customized.getNumPartitions()
n_items_customized = top10_reviewed_customized.mapPartitions(lambda p: [sum(1 for i in p)]).collect()
end_time_customized = time.time()

with open('task2.json', 'w') as f:
    json.dump({"default": {'n_partition': n_partition, 'n_items': n_items, 'exe_time': end_time - start_time}, "customized": {'n_partition': n_partition_customized, 'n_items': n_items_customized, 'exe_time': end_time_customized - start_time_customized}}, f)

print("Default partition: ", n_partition, n_items, end_time - start_time)
print("Customized partition: ", n_partition_customized, n_items_customized, end_time_customized - start_time_customized)