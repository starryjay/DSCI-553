import os

os.environ['PYSPARK_SUBMIT_ARGS'] = '--packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 pyspark-shell'

import pyspark
import sys
from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import time

# run
# spark-submit --packages graphframes:graphframes:0.8.2-spark3.1-s_2.12 task1.py 
# <filter threshold> <input_file_path> <community_output_file_path>

sc = pyspark.SparkContext('local[*]')
spark = SparkSession(sc)
spark.sparkContext.setLogLevel("ERROR")
filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
output_file_path = sys.argv[3]

def process_data(input_path):
    data = spark.read.csv(input_path, header=True, inferSchema=True)
    return data


def graph_construction(df):
    # each node (user) is uniquely labeled
    # links are undirected and unweighted
    # there should be an edge between two nodes if the number of common businesses between them is >= the filter threshold
    # if there is no edge connected to a particular node, that node should not be present in the graph
    # the filter threshold is an input parameter given by filter_threshold
    business_sets = df.groupBy('user_id').agg(F.collect_set('business_id').alias('business_set'))
    # construct pairs of users
    pairs = business_sets.join(business_sets.withColumnRenamed('user_id', 'user_id2').withColumnRenamed('business_set', 'business_set2')).filter('user_id != user_id2')
    # filter pairs with common businesses >= filter_threshold
    pairs = pairs.withColumn('common_businesses', F.size(F.array_intersect('business_set', 'business_set2')))
    pairs = pairs.filter(pairs.common_businesses >= filter_threshold)
    # rename to src and dst
    edges = pairs.select(F.least('user_id', 'user_id2').alias('src'), F.greatest('user_id', 'user_id2').alias('dst')).distinct()
    edges2 = pairs.select(F.greatest('user_id', 'user_id2').alias('src'), F.least('user_id', 'user_id2').alias('dst')).distinct()
    edges = edges.union(edges2).distinct()
    sourcenodes = edges.select('src').distinct().withColumnRenamed('src', 'id')
    destnodes = edges.select('dst').distinct().withColumnRenamed('dst', 'id')
    nodes = sourcenodes.union(destnodes).distinct()
    # construct graph
    graph = GraphFrame(nodes, edges)
    print(graph)
    return graph

def community_detection(graph):
    comms = graph.labelPropagation(maxIter=5)
    return comms

# save output in txt file where each line is a community
# format is 'user1', 'user2', 'user3', ...
# sort by size of communities ascending, then by lexicographical order of first user
# communities should also be sorted in lexicographical order of users
def save_output(communities, output_path):
    communities = communities.select('label', 'id').rdd
    communities = communities.map(lambda x: (x[0], x[1])).groupByKey().mapValues(list).map(lambda x: (x[0], sorted(x[1]))).sortBy(lambda x: (len(x[1]), x[1][0])).collect()
    #communities = communities.groupBy('label').agg(F.collect_list('id').alias('users')).orderBy([F.size('users'), 'users'], ascending=[True, True]).collect()
    with open(output_path, 'w') as f:
        for comm in communities:
            users = sorted(map(str, comm[1]))
            f.write("\'" + '\', \''.join(users) +  "\'" + '\n')

def main():
    df = process_data(input_file_path)
    graph = graph_construction(df)
    communities = community_detection(graph)
    save_output(communities, output_file_path)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Duration:', end - start)