import pyspark
from collections import defaultdict
import sys
import math
from itertools import combinations

sc = pyspark.SparkContext('local[*]')
sc.setLogLevel("ERROR")

case = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]

data = sc.textFile(input_file_path).map(lambda row: row.split(','))
header = data.first()
data = data.filter(lambda row: row != header)

def SON_pass1(data, support):
    if case == 1:
        baskets = data.map(lambda row: (row[0], [row[1]])).reduceByKey(lambda u1, u2: u1 + u2).map(lambda row: set(row[1]))
    elif case == 2:
        baskets = data.map(lambda row: (row[1], [row[0]])).reduceByKey(lambda b1, b2: b1 + b2).map(lambda row: set(row[1]))
    p = baskets.getNumPartitions()
    candidates = baskets.mapPartitions(lambda partition: apriori(partition, support, p)).sortBy(lambda itemset: (len(itemset[0]), itemset[0]))
    return candidates, baskets

def apriori(baskets, support, p):
    items_dict = defaultdict(int)
    baskets = list(baskets)
    scaled_support = support / p
    for basket in baskets:
        for item in basket:
            items_dict[(item,)] += 1
    cands = {item: count for item, count in items_dict.items() if count >= scaled_support}
    freqs = cands if len(cands) > 0 else {}
    k = 2
    while True:
        items_dict = defaultdict(int)
        for basket in baskets:
            existing_itemsets = sorted(list(set(basket).intersection(cands)))
            for item in combinations(existing_itemsets, k):
                items_dict[item] += 1
        cands = {item: count for item, count in items_dict.items() if count >= scaled_support}
        if len(cands) > 0:
            freqs.update(cands)
        else:
            break
        cands = set([itemset for basket in cands for itemset in basket])
        k += 1
    return list(freqs.items())

    # support_scaled = support / p
    # items_dict = defaultdict(int)
    # frequents_dict = defaultdict(int)
    # baskets = list(baskets)
    # for basket in baskets:
    #     for item in set(basket):
    #         items_dict[(item,)] += 1
    #         if items_dict[(item,)] >= support_scaled:
    #             frequents_dict[(item,)] = items_dict[(item,)]
    # k = 2
    # current_freqs = set(frequents_dict.keys())
    # while current_freqs:
    #     candidates = set()
    #     for x in current_freqs:
    #         for y in current_freqs:
    #             joined = set(x).union(set(y))
    #             if len(joined) == k:
    #                 candidates.add(frozenset(joined))
    #     candidate_counts = defaultdict(int)
    #     for basket in baskets:
    #         for candidate in candidates:
    #             if set(candidate).issubset(set(basket)):
    #                 candidate_counts[candidate] += 1
    #     current_freqs = set()
    #     for candidate, count in candidate_counts.items():
    #         if count >= support_scaled:
    #             current_freqs.add(candidate)
    #             frequents_dict[candidate] += count
    #     k += 1
    # return [(tuple(itemset), count) for itemset, count in frequents_dict.items()]

def SON_pass2(candidates, baskets, support):
    cands = candidates.collectAsMap()
    candidate_count = baskets.flatMap(lambda basket: [(candidate, 1) for candidate in cands.keys() if set(candidate).issubset(set(basket))]).reduceByKey(lambda a, b: a + b)
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
    
    # remove last two newlines
    with open(output_file_path, 'rb+') as f:
        f.seek(-2, 2)
        f.truncate()

if __name__ == "__main__":
    candidates, baskets = SON_pass1(data, support)
    frequent_itemsets = SON_pass2(candidates, baskets, support)
    write_to_file(candidates.collect(), frequent_itemsets)
    sc.stop()