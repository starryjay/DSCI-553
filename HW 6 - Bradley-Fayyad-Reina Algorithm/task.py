import numpy as np
from sklearn.cluster import KMeans
import sys
import time
import random

np.random.seed(42)
random.seed(42)

input_file = sys.argv[1]
n_cluster = int(sys.argv[2])
output_file = sys.argv[3]

def load_data(input_file):
    with open(input_file, 'r') as f:
        data = np.array(f.readlines())
    data = np.array([np.array(list(map(float, x.strip().split(',')))) for x in data])
    np.random.shuffle(data)
    data_split = np.array_split(data, 5)
    return data_split

def bfr_init(data_split, n_cluster):
    dim = data_split[0].shape[1] - 2
    ds_clusters = {}
    cs_clusters = {}
    rs_points = []
    int_res = []
    init_data = data_split[0]
    kmeans = KMeans(n_clusters=n_cluster * 5).fit(init_data[:, 2:])
    labels = kmeans.labels_
    cluster_points = {}
    for index, cluster_id in enumerate(labels):
        cluster_points.setdefault(cluster_id, []).append(index)
    for points in cluster_points.values():
        if len(points) == 1:
            rs_points.append(init_data[points[0]])
    ds_indices = [index for points in cluster_points.values() if len(points) > 1 for index in points]
    ds_data = init_data[ds_indices]
    ds_ids = ds_data[:, 0].astype(int)
    ds_features = ds_data[:, 2:]
    kmeans = KMeans(n_clusters=n_cluster).fit(ds_features)
    labels = kmeans.labels_
    for index, cluster_id in enumerate(labels):
        cluster = ds_clusters.setdefault(cluster_id, {'N': 0, 'SUM': np.zeros(dim), 'SUMSQ': np.zeros(dim), 'ids': []})
        cluster['N'] += 1
        cluster['SUM'] += ds_features[index]
        cluster['SUMSQ'] += ds_features[index] ** 2
        cluster['ids'].append(ds_ids[index])
    if len(rs_points) >= n_cluster * 5:
        rs_features = np.array(rs_points)[:, 2:]
        kmeans = KMeans(n_clusters=n_cluster * 5).fit(rs_features)
        labels = kmeans.labels_
        cluster_points = {}
        for index, cluster_id in enumerate(labels):
            cluster_points.setdefault(cluster_id, []).append(index)
        new_rs_points = []
        for points in cluster_points.values():
            if len(points) == 1:
                new_rs_points.append(rs_points[points[0]])
            else:
                cluster = {'N': 0, 'SUM': np.zeros(dim), 'SUMSQ': np.zeros(dim), 'ids': []}
                for index in points:
                    point = rs_points[index]
                    cluster['N'] += 1
                    cluster['SUM'] += point[2:]
                    cluster['SUMSQ'] += point[2:] ** 2
                    cluster['ids'].append(int(point[0]))
                cs_clusters[len(cs_clusters)] = cluster
        rs_points = new_rs_points
    num_discard = sum([cluster['N'] for cluster in ds_clusters.values()])
    num_compress = sum([cluster['N'] for cluster in cs_clusters.values()])
    int_res.append(f"Round 1: {num_discard},{len(cs_clusters)},{num_compress},{len(rs_points)}\n")
    return ds_clusters, cs_clusters, rs_points, int_res

def bfr_loop(data_split, n_cluster, ds_clusters, cs_clusters, rs_points, int_res):  
    dim = data_split[0].shape[1] - 2
    d = 2 * np.sqrt(dim)
    for i in range(1, 5):
        data = data_split[i]
        ids = data[:, 0].astype(int)
        unassigned = []
        for index in range(len(data)):
            point_id = ids[index]
            point = data[:, 2:][index]
            min_dist = float('inf')
            closest_cluster = None
            for cluster_id, cluster in ds_clusters.items():
                mean = cluster['SUM'] / cluster['N']
                variance = (cluster['SUMSQ'] / cluster['N']) - (mean ** 2)
                variance[variance == 0] = 1e-10
                std_dev = np.sqrt(variance)
                mh_dist = np.sqrt(np.sum(((point - mean) / std_dev) ** 2))
                if mh_dist < min_dist:
                    min_dist = mh_dist
                    closest_cluster = cluster_id
            if min_dist < d:
                cluster = ds_clusters[closest_cluster]
                cluster['N'] += 1
                cluster['SUM'] += point
                cluster['SUMSQ'] += point ** 2
                cluster['ids'].append(point_id)
            else:
                min_dist = float('inf')
                closest_cluster = None
                for cluster_id, cluster in cs_clusters.items():
                    mean = cluster['SUM'] / cluster['N']
                    variance = (cluster['SUMSQ'] / cluster['N']) - (mean ** 2)
                    variance[variance == 0] = 1e-10
                    std_dev = np.sqrt(variance)
                    mh_dist = np.sqrt(np.sum(((point - mean) / std_dev) ** 2))
                    if mh_dist < min_dist:
                        min_dist = mh_dist
                        closest_cluster = cluster_id
                if min_dist < d:
                    cluster = cs_clusters[closest_cluster]
                    cluster['N'] += 1
                    cluster['SUM'] += point
                    cluster['SUMSQ'] += point ** 2
                    cluster['ids'].append(point_id)
                else:
                    unassigned.append(data[index])
        rs_points.extend(unassigned)

        if len(rs_points) >= n_cluster * 5:
            rs_features = np.array(rs_points)[:, 2:]
            kmeans = KMeans(n_clusters=n_cluster * 5).fit(rs_features)
            labels = kmeans.labels_
            cluster_points = {}
            for index, cluster_id in enumerate(labels):
                cluster_points.setdefault(cluster_id, []).append(index)
            new_rs_points = []
            for points in cluster_points.values():
                if len(points) == 1:
                    new_rs_points.append(rs_points[points[0]])
                else:
                    cluster = {'N': 0, 'SUM': np.zeros(dim), 'SUMSQ': np.zeros(dim), 'ids': []}
                    for index in points:
                        point = rs_points[index]
                        cluster['N'] += 1
                        cluster['SUM'] += point[2:]
                        cluster['SUMSQ'] += point[2:] ** 2
                        cluster['ids'].append(int(point[0]))
                    cs_clusters[len(cs_clusters)] = cluster
            rs_points = new_rs_points
        
        cs_labels = list(cs_clusters.keys())
        merged = True
        while merged:
            merged = False
            for i in range(len(cs_labels)):
                for j in range(i + 1, len(cs_labels)):
                    label1 = cs_labels[i]
                    label2 = cs_labels[j]
                    cluster1 = cs_clusters[label1]
                    cluster2 = cs_clusters[label2]
                    mean1 = cluster1['SUM'] / cluster1['N']
                    variance1 = (cluster1['SUMSQ'] / cluster1['N']) - (mean1 ** 2)
                    variance1[variance1 == 0] = 1e-10
                    mean2 = cluster2['SUM'] / cluster2['N']
                    variance2 = (cluster2['SUMSQ'] / cluster2['N']) - (mean2 ** 2)
                    variance2[variance2 == 0] = 1e-10
                    std_dev1 = np.sqrt(variance1)
                    std_dev2 = np.sqrt(variance2)
                    mh_dist1 = np.sqrt(np.sum(((mean1 - mean2) / std_dev2) ** 2))
                    mh_dist2 = np.sqrt(np.sum(((mean2 - mean1) / std_dev1) ** 2))
                    mh_dist = min(mh_dist1, mh_dist2)
                    if mh_dist < d and mh_dist == mh_dist1:
                        cluster2['N'] += cluster1['N']
                        cluster2['SUM'] += cluster1['SUM']
                        cluster2['SUMSQ'] += cluster1['SUMSQ']
                        cluster2['ids'].extend(cluster1['ids'])
                        cs_clusters.pop(label1)
                        cs_labels.remove(label1)
                        merged = True
                        break
                    elif mh_dist < d and mh_dist == mh_dist2:
                        cluster1['N'] += cluster2['N']
                        cluster1['SUM'] += cluster2['SUM']
                        cluster1['SUMSQ'] += cluster2['SUMSQ']
                        cluster1['ids'].extend(cluster2['ids'])
                        cs_clusters.pop(label2)
                        cs_labels.remove(label2)
                        merged = True
                        break
                if merged:
                    break
        num_discard = sum([cluster['N'] for cluster in ds_clusters.values()])
        num_compress = sum([cluster['N'] for cluster in cs_clusters.values()])
        int_res.append(f"Round {i + 1}: {num_discard},{len(cs_clusters)},{num_compress},{len(rs_points)}\n")
    
    for cluster in cs_clusters.values():
        min_dist = float('inf')
        closest_cluster = None
        mean_cs = cluster['SUM'] / cluster['N']
        for cluster_id, ds_cluster in ds_clusters.items():
            mean_ds = ds_cluster['SUM'] / ds_cluster['N']
            variance_ds = (ds_cluster['SUMSQ'] / ds_cluster['N']) - (mean_ds ** 2)
            variance_ds[variance_ds == 0] = 1e-10
            std_dev = np.sqrt(variance_ds)
            mh_dist = np.sqrt(np.sum(((mean_cs - mean_ds) / std_dev) ** 2))
            if mh_dist < min_dist:
                min_dist = mh_dist
                closest_cluster = cluster_id
        if min_dist < d:
            ds_cluster = ds_clusters[closest_cluster]
            ds_cluster['N'] += cluster['N']
            ds_cluster['SUM'] += cluster['SUM']
            ds_cluster['SUMSQ'] += cluster['SUMSQ']
            ds_cluster['ids'].extend(cluster['ids'])
        else:
            ds_clusters[len(ds_clusters)] = cluster
    
    for point in rs_points:
        point_id = int(point[0])
        point_features = point[2:]
        min_dist = float('inf')
        closest_cluster = None
        for cluster_id, cluster in ds_clusters.items():
            mean = cluster['SUM'] / cluster['N']
            variance = (cluster['SUMSQ'] / cluster['N']) - (mean ** 2)
            variance[variance == 0] = 1e-10
            std_dev = np.sqrt(variance)
            mh_dist = np.sqrt(np.sum(((point_features - mean) / std_dev) ** 2))
            if mh_dist < min_dist:
                min_dist = mh_dist
                closest_cluster = cluster_id
        if min_dist < d:
            cluster = ds_clusters[closest_cluster]
            cluster['N'] += 1
            cluster['SUM'] += point_features
            cluster['SUMSQ'] += point_features ** 2
            cluster['ids'].append(point_id)
    return ds_clusters, rs_points, int_res
    
def write_output(output_file, ds_clusters, rs_points, int_res):
    with open(output_file, 'w') as f:
        f.write("The intermediate results:\n")
        f.writelines(int_res)
        f.write("\nThe clustering results:\n")
        clustering_results = {}
        for cluster_id, cluster in ds_clusters.items():
            for point_id in cluster['ids']:
                clustering_results[point_id] = cluster_id
        for point in rs_points:
            point_id = int(point[0])
            clustering_results[point_id] = -1
        for point_id in sorted(clustering_results.keys()):
            f.write(f"{point_id},{clustering_results[point_id]}\n")

def main():
    data_split = load_data(input_file)
    ds_clusters, cs_clusters, rs_points, int_res = bfr_init(data_split, n_cluster)
    with open('rs_points_rd1.txt', 'w') as f:
        for point in rs_points:
            f.write(','.join(map(str, point)) + '\n')
    ds_clusters, rs_points, int_res = bfr_loop(data_split, n_cluster, ds_clusters, cs_clusters, rs_points, int_res)
    write_output(output_file, ds_clusters, rs_points, int_res)

if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(f"Duration: {end - start}")
