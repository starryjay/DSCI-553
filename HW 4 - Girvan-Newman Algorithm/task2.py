import sys
import pyspark
import time
from collections import defaultdict

# dont initialize edge weights of non-leaf nodes to 0

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
            for child in graph[curr]:
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
                    num_paths[child] += num_paths[curr]
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

def calculate_modularity(graph, communities, original_degrees, m):
    modularity = 0
    for community in communities:
        for i in community:
            for j in community:
                Aij = 1 if j in graph[i] else 0
                modularity += Aij - (original_degrees[i] * original_degrees[j]) / (2 * m)
    modularity /= 2 * m
    return modularity

def girvan_newman(adjacency_matrix):
    original_graph = adjacency_matrix.collectAsMap()
    m = sum(len(neighbors) for neighbors in original_graph.values()) // 2
    original_degrees = {node: len(neighbors) for node, neighbors in original_graph.items()}
    best_modularity = -1
    best_communities = None
    adjacency_matrix = adjacency_matrix.collectAsMap()
    while len(adjacency_matrix) > 0:
        betweenness_scores = betweenness(sc.parallelize(adjacency_matrix.items()))
        max_betweenness = betweenness_scores[0][1]
        edges_to_remove = [edge for edge, score in betweenness_scores if score == max_betweenness]
        for edge in edges_to_remove:
            node1, node2 = edge
            if node1 in adjacency_matrix.keys(): 
                if node2 in adjacency_matrix[node1]:
                    adjacency_matrix[node1].remove(node2)
                if not adjacency_matrix[node1]: del adjacency_matrix[node1]
            if node2 in adjacency_matrix.keys():
                if node1 in adjacency_matrix[node2]:
                    adjacency_matrix[node2].remove(node1)
                if not adjacency_matrix[node2]: del adjacency_matrix[node2]
        communities = find_communities(adjacency_matrix)
        current_modularity = calculate_modularity(adjacency_matrix, communities, original_degrees, m)
        if current_modularity > best_modularity:
            best_modularity = current_modularity
            best_communities = communities
    return best_communities

def find_communities(adjacency_matrix):
    visited = set()
    communities = []
    for node in adjacency_matrix:
        if node not in visited:
            community = set()
            queue = [node]
            visited.add(node)
            while queue:
                current = queue.pop(0)
                community.add(current)
                for neighbor in adjacency_matrix[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            communities.append(list(community))
    return communities

def sort_communities(communities):
    for i in range(len(communities)):
        communities[i] = sorted(communities[i])
    communities = sorted(communities, key=lambda x: (len(x), x)) # why isn't this working
    return communities

def write_communities(communities, output_path):
    with open(output_path, 'w') as f:
        for community in communities:
            f.write('\'' + community[0] + '\'')
            for node in community[1:]:
                f.write(', \'' + node + '\'')
            f.write('\n')

def main():
    rdd = process_data(input_file_path)
    adjacency_matrix = graph_construction(rdd)
    between = betweenness(adjacency_matrix)
    write_betweenness(between, betweenness_file_path)
    communities = girvan_newman(adjacency_matrix)
    communities = sort_communities(communities)
    write_communities(communities, community_file_path)

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print('Duration:', end - start)
    sc.stop()