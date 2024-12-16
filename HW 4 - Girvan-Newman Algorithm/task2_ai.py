import sys
import pyspark
import time
from collections import defaultdict

sc = pyspark.SparkContext('local[*]')
filter_threshold = int(sys.argv[1])
input_file_path = sys.argv[2]
betweenness_file_path = sys.argv[3]
community_file_path = sys.argv[4]

def process_data(input_path):
    data = sc.textFile(input_path).map(lambda x: x.split(','))
    header = data.first()
    data = data.filter(lambda x: x != header)
    return data

def graph_construction(rdd):
    business_sets = rdd.map(lambda row: (row[0], row[1])).groupByKey().mapValues(set)
    pairs = business_sets.cartesian(business_sets).filter(lambda x: x[0][0] != x[1][0])
    pairs = pairs.map(lambda x: (x[0][0], x[1][0], len(x[0][1].intersection(x[1][1]))))
    pairs = pairs.filter(lambda x: x[2] >= filter_threshold)
    edges = pairs.map(lambda x: (x[0], x[1])).distinct()
    edges2 = pairs.map(lambda x: (x[1], x[0])).distinct()
    edges = edges.union(edges2).distinct()
    adjacency_matrix = edges.groupByKey().mapValues(list)
    return adjacency_matrix

def betweenness(adjacency_matrix):

    def bfs(graph, node):
        # graph = {node: [neighbors]}
        # node = starting node (str)
        parent = defaultdict(set)
        level = {node: 0}
        num_paths = defaultdict(int)
        num_paths[node] = 1
        traversal = []
        queue = [node]
        visited = {node}
        while queue:
            curr = queue.pop(0)
            traversal.append(curr)
            for child in graph[curr]: # [neighbors]
                if child not in visited: 
                    visited.add(child) # {child1, child2, ...}
                    queue.append(child) # [child1, child2, ...]
                    num_paths[child] += num_paths[curr] # {child1: 1, child2: 1, ...}
                    level[child] = level[curr] + 1 
                    parent[child].add(curr)
                elif level[child] == level[curr] + 1:
                    num_paths[child] += num_paths[curr]
                    parent[child].add(curr)
        return parent, traversal, num_paths

    def calculate_betweenness(graph, node):
        parent, traversal, num_paths = bfs(graph, node)
        edge_credit = defaultdict(float)
        node_credit = defaultdict(float)
        for i in range(len(traversal) - 1, -1, -1):
            curr = traversal[i]
            for p in parent[curr]:
                edge_credit[(curr, p)] = num_paths[p] / num_paths[curr] * (1 + node_credit[curr])
                node_credit[p] += edge_credit[(curr, p)]
        return edge_credit
    
    adjacency_matrix = adjacency_matrix.collectAsMap()
    betweenness = defaultdict(float)
    for node in adjacency_matrix:
        edge_credit = calculate_betweenness(adjacency_matrix, node)
        for edge in edge_credit:
            betweenness[edge] += edge_credit[edge]
    betweenness = sorted(betweenness.items(), key=lambda x: (-x[1], x[0]))
    return betweenness

def write_betweenness(betweenness, output_path):
    with open(output_path, 'w') as f:
        for edge in betweenness:
            if edge[0][0] < edge[0][1]:
                f.write('(\'' + edge[0][0] + '\', \'' + edge[0][1] + '\'),' + str(round(edge[1], 5)) + '\n')

def girvan_newman(adjacency_matrix, betweenness):
    betweenness = [edge[0] for edge in betweenness]
    highest_modularity = float('-inf')
    best_communities = []

    def connected_components(graph):
        visited = set()
        components = []
        for node in graph.keys():
            if node not in visited:
                component = []
                queue = [node]
                visited.add(node)
                while queue:
                    curr_node = queue.pop(0)
                    component.append(curr_node)
                    for neighbor in graph[curr_node]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                            visited.add(neighbor)
                components.append(component)
        return components
    
    original_adjacency_matrix = adjacency_matrix.mapValues(set).collectAsMap()
    adjacency_matrix = adjacency_matrix.collectAsMap()
    while betweenness:
        edge = betweenness.pop(0)
        if edge[0] not in adjacency_matrix[edge[1]] or edge[1] not in adjacency_matrix[edge[0]]:
            continue
        adjacency_matrix[edge[0]].remove(edge[1])
        adjacency_matrix[edge[1]].remove(edge[0])
        modularity = 0
        total_edges = sum([len(adjacency_matrix[node]) for node in adjacency_matrix]) // 2
        if total_edges == 0: 
            continue
        for community in connected_components(adjacency_matrix):
            for node1 in community:
                for node2 in community:
                    A = 1 if ((node2 in original_adjacency_matrix[node1]) and (node1 in original_adjacency_matrix[node2])) else 0
                    modularity += A - (len(original_adjacency_matrix[node1]) * len(original_adjacency_matrix[node2]) / (2 * total_edges))
        modularity /= (2 * total_edges)
        if modularity > highest_modularity:
            highest_modularity = modularity
            best_communities = connected_components(adjacency_matrix)
    return best_communities

def sort_communities(communities):
    communities = sorted(communities, key=lambda x: (len(x), x))
    for i in range(len(communities)):
        communities[i] = sorted(communities[i])
    print('Communities:', communities)
    return communities

def write_communities(communities, output_path):
    with open(output_path, 'w') as f:
        for community in communities:
            f.write('\'' + community[0] + '\'')
            for node in community[1:]:
                f.write(', \'' + node + '\'')
            f.write('\n')

if __name__ == '__main__':
    start = time.time()
    rdd = process_data(input_file_path)
    adjacency_matrix = graph_construction(rdd)
    between = betweenness(adjacency_matrix)
    write_betweenness(between, betweenness_file_path)
    communities = girvan_newman(adjacency_matrix, between)
    communities = sort_communities(communities)
    write_communities(communities, community_file_path)
    end = time.time()
    print('Duration:', end - start)
    sc.stop()