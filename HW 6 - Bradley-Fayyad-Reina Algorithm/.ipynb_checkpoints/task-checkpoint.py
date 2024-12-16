import numpy as np
from sklearn.cluster import KMeans
import pyspark
import time
import sys
import os

# number of dimensions in testing dataset could be different
# from hw6_clustering.txt

# Bradley-Fayyad-Reina (BFR) algorithm

# 3 sets of points to keep track of:
    # 1. Discard set (DS)
    # 2. Compression set (CS)
    # 3. Retained set (RS)

# Each cluster is summarized by N (number of points), 
# SUM (sum of coordinates), and 
# SUMSQ (sum of squares of coordinates)

# Step 1. Load 20% of the data randomly
# Step 2: Run K-means from sklearn with a large K, e.g.
        # 5x the number of input clusters on the data in memory,
        # using Euclidean distance as the similarity measure.
# Step 3: In the K-means result from Step 2, move all the
        # clusters that contain only one point to the RS
# Step 4: Run K-means again to cluster the rest of the data
        # with K = the number of input clusters.
# Step 5: Use the K-means result from Step 4 to generate the
        # DS clusters, i.e. discard their points and generate
        # the statistics for the DS clusters.
# Step 6: Run K-means on the points in the RS with a large K
        # to generate the CS clusters (>1 points) and RS (1 point).
# Step 7: Load another 20% of the data randomly.
# Step 8: For the new points, compare them to each of the DS
        # clusters using Mahalanobis distance. Assign them to
        # the nearest DS clusters if the distance is < 2(sqrt(d)).
# Step 9: For new points not assigned to DS clusters, assign
        # them to the nearest CS clusters if Mahalanobis distance
        # is < 2(sqrt(d)).
# Step 10: For new points not assigned to any DS or CS clusters,
        # assign them to RS.
# Step 11: Run K-means on the points in the RS with a large K.
# Step 12: Merge CS clusters that have a Mahalanobis distance
        # < 2(sqrt(d)).
# Repeat steps 7-12 until all data points are processed.
# If this is the last run/after the last chunk of data,
# merge the CS clusters with the DS clusters that have a 
# Mahalanobis distance < 2(sqrt(d)).

# At each run, including initialization, you need to count
# and ouput the number of DS points, number of clusters in CS,
# number of CS points, and number of RS points.

# Input format: input_file, n_cluster, output_file.
# Output file is a text file containing:
    # Intermediate results. First line is "The intermediate results".
    # Then each line starts with "Round {i}:" and i is the count
    # for the round. Output order is DS points, CS clusters, CS points,
    # RS points.
    # Skip a line after all intermediate results are printed.
    # Clustering results. First line is "The clustering results".
    # Include the data points index (first column of input file)
    # and the clustering result after the BFR algorithm.
    # Cluster of outliers should be represented as -1.
# Percentage of discard points after the last round should be >=98.5%.
# Accuracy = (# of pts in correct clusters) / (# of pts in ground truth)
# Runtime < 10 minutes.

input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

sc = pyspark.SparkContext()
os.environ['PYSPARK_PYTHON'] = '/Users/shobhanashreedhar/anaconda3/envs/datamining/bin/python'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/shobhanashreedhar/anaconda3/envs/datamining/bin/python'

def load_data(input_file):
    data = sc.textFile(input_file).map(lambda x: x.split(sep=',')).map(lambda x: [float(i) for i in x]).map(lambda x: x[2:])
    return data

def mahalanobis_distance(x, y, cov):
    return np.sqrt(np.dot(np.dot((x-y).T, np.linalg.inv(cov)), (x-y)))

def bfr_init(data, n_cluster):
    data = data.sample(withReplacement=False, fraction=0.2, seed=553)
    data_array = np.array(data.collect())
    print('data array:', data_array)
    print('data array shape:', data_array.shape)
    kmeans = KMeans(n_clusters=5*n_cluster, random_state=553).fit(data_array)
    rs = sc.parallelize(kmeans.cluster_centers_[np.unique(kmeans.labels_, return_counts=True)[1] == 1])
    kmeans = KMeans(n_clusters=n_cluster, random_state=553).fit(data_array)
    ds = sc.parallelize(kmeans.cluster_centers_)
    return ds, rs

def bfr(ds, rs, n_cluster):
    rs_array = np.array(rs.collect())
    if len(rs_array) < 5 * n_cluster:
        k = len(rs_array)
    else:
        k = 5 * n_cluster
    kmeans = KMeans(n_clusters=k, random_state=553).fit(rs_array)
    cs = sc.parallelize(kmeans.cluster_centers_[np.unique(kmeans.labels_, return_counts=True)[1] > 1])
    rs = sc.parallelize(kmeans.cluster_centers_[np.unique(kmeans.labels_, return_counts=True)[1] == 1])
    return cs, rs

def main():
        data = load_data(input_file)
        ds, rs = bfr_init(data, n_cluster)
        cs, rs = bfr(ds, rs, n_cluster)
        print("The intermediate results")
        print("Round 1:", ds.count(), cs.count(), cs.count(), rs.count())

if __name__ == "__main__":
        start = time.time()
        main()
        end = time.time()
        print("Time taken: ", end-start)
