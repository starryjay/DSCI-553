import pyspark
from collections import defaultdict
import sys
import time
from itertools import combinations

sc = pyspark.SparkContext('local[*]')
filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

data = sc.textFile(input_file_path).map(lambda row: row.split(','))
header = data.first()
data = data.filter(lambda row: row != header)

def preprocessing(data):
    preprocessed_data = data.map(lambda row: (str(row[0][1:6]) + str(row[0][-3:-1]) + '-' + str(int(row[1][1:-1])), str(int(row[5][1:-1])))).map(lambda row: (row[0], row[1]))
    return preprocessed_data

def filter_data(data, filter_threshold):
    data = data.map(lambda row: (row[0], [row[1]])).reduceByKey(lambda u1, u2: u1 + u2).map(lambda row: (row[0], tuple(row[1]))).filter(lambda row: len(row[1]) > filter_threshold)
    return data

def SON_pass1(data, support):
    baskets = data.map(lambda row: (row[0], [*row[1]])).reduceByKey(lambda u1, u2: u1 + u2).map(lambda row: set(row[1]))
    p = baskets.getNumPartitions()
    candidates = baskets.mapPartitions(lambda partition: apriori(partition, support, p)).distinct().sortBy(lambda itemset: (len(itemset), itemset))
    return candidates, baskets

def apriori(baskets, support, p):
    support_scaled = support / p
    items_dict = defaultdict(int)
    frequents_dict = defaultdict(int)
    baskets = list(baskets)
    for basket in baskets:
        for item in basket:
            items_dict[(item,)] += 1
            if items_dict[(item,)] >= support_scaled:
                frequents_dict[(item,)] = items_dict[(item,)]
    k = 2
    current_freqs = frequents_dict
    while current_freqs:
        candidates = set()
        combs = combinations(current_freqs, 2)
        for x in current_freqs:
            for y in current_freqs:
                joined = set(x).union(set(y))
                if len(joined) == k:
                    current_freqs[k].add(frozenset(joined))
        candidate_counts = defaultdict(int)
        for basket in baskets:
            for candidate in candidates:
                if set(candidate).issubset(set(basket)):
                    candidate_counts[candidate] += 1
        current_freqs = set()
        for candidate, count in candidate_counts.items():
            if count >= support_scaled:
                current_freqs.add(candidate)
                frequents_dict[candidate] += count
        k += 1
    return [(tuple(itemset), count) for itemset, count in frequents_dict.items()]

def SON_pass2(candidates, baskets, support):
    candidate_itemsets = candidates.map(lambda x: x[0]).collect()
    candidate_count = baskets.flatMap(lambda basket: [(candidate, 1) for candidate in candidate_itemsets if set(candidate).issubset(basket)]).reduceByKey(lambda a, b: a + b)
    frequent_itemsets = candidate_count.filter(lambda x: x[1] >= support).sortBy(lambda x: (len(x[0]), x[0])).collect()
    return frequent_itemsets

def write_to_file(candidates, frequent_itemsets):
    def group_output(f, itemsets):
        grouped_itemsets = defaultdict(list)
        for itemset, _ in itemsets:
            if len(itemset) == 1:
                grouped_itemsets[1].append(f"('{itemset[0]}')")
                grouped_itemsets[1] = sorted(list(set(grouped_itemsets[1])))
            elif len(itemset) > 1:
                grouped_itemsets[len(itemset)].append(str(tuple(sorted(itemset))))
                grouped_itemsets[len(itemset)] = sorted(list(set(grouped_itemsets[len(itemset)])))
        
        for length in sorted(grouped_itemsets.keys()):
            f.write(','.join(sorted(grouped_itemsets[length])))
            f.write('\n\n')

    with open(output_file_path, "w") as f:
        f.write('Candidates:\n')
        group_output(f, candidates)

    with open(output_file_path, "a") as f:
        f.write('Frequent Itemsets:\n')
        group_output(f, frequent_itemsets)

    with open(output_file_path, 'rb+') as f:
        f.seek(-2, 2)
        f.truncate()


if __name__ == "__main__":
    start = time.time()
    preprocessed_data = preprocessing(data)
    data_new = filter_data(preprocessed_data, filter_threshold)
    data_new.cache()
    candidates, baskets = SON_pass1(data_new, support)
    candidates.cache()
    baskets.cache()
    frequent_itemsets = SON_pass2(candidates, baskets, support)
    write_to_file(candidates.collect(), frequent_itemsets)
    end = time.time()
    print('Duration:', end - start)
    sc.stop()