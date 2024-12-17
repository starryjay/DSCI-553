import pyspark
import random
from itertools import combinations
import sys
import time

random.seed(39)

start = time.time()

input_file = sys.argv[1]
output_file = sys.argv[2]

sc = pyspark.SparkContext().getOrCreate()
sc.setLogLevel("WARN")

data = sc.textFile(input_file).map(lambda row: row.split(','))

header = data.first()
data = data.filter(lambda row: row != header)

user_dict = data.map(lambda row: row[0]).distinct().sortBy(lambda x: x).zipWithIndex().collectAsMap() # user_id -> index

data_grouped = data.map(lambda row: (row[1], row[0])).map(lambda x: (x[0], user_dict[x[1]])).groupByKey().mapValues(list) # (business_id, [user_id1, user_id2, ...])
business_user_dict = data_grouped.collectAsMap() # {business_id1: [user_id1, user_id2, ...], ...}

num_hashes = 300
rows_per_band = 4
num_bands = num_hashes // rows_per_band
prime = 1013501
hash_functions = []

def generate_hash(num_hashes, prime):
    a = random.randint(1, prime - 1)
    b = random.randint(1, prime - 1)
    return lambda x: ((a * x + b) % prime) % num_hashes

for i in range(num_hashes):
    hash_functions.append(generate_hash(num_hashes, prime))

                                                # business_id,   smallest hash value for each hash function
minhash_signatures = data_grouped.map(lambda row: (row[0], [min([hash_func(idx) for idx in row[1]]) for hash_func in hash_functions])) # (business_id, [sig1, sig2, ...])

len_minhash_signatures = minhash_signatures.count()

                                                # band number, signature for that band                                  business_id                      # group by band and signature to find candidate pairs of businesses
bands = minhash_signatures.flatMap(lambda row: [((band, tuple(row[1][band * rows_per_band:(band + 1) * rows_per_band])), row[0]) for band in range(num_bands)]).groupByKey().mapValues(list)
# ((band, signature), [business_id]) <-- group business ids by matching bands and signatures

                            # make sure cands have at least 2 business_ids    # get all pairs of business_ids in each band             # (pair, number of bands)     # get only the pairs
candidates = bands.filter(lambda row: len(row[1]) > 1).flatMap(lambda row: [(pair, 1) for pair in combinations(row[1], 2)]).reduceByKey(lambda b1, b2: b1 + b2).map(lambda x: x[0]) # lists of business_ids

def jaccard(a, b):
    a = set(a)
    b = set(b)
    return len(a.intersection(b)) / len(a.union(b))

                                            # get pairs in lexicographical order and jaccard similarity of users per business pair  # filter pairs with jaccard similarity >= 0.5  # sort by business_id1 and business_id2
final_pairs = candidates.map(lambda pair: (sorted(pair)[0], sorted(pair)[1], 
                                           jaccard(business_user_dict[pair[0]], business_user_dict[pair[1]]))).filter(lambda row: row[2] >= 0.5).sortBy(lambda row: [row[0], row[1]])
# (business_id1, business_id2, similarity)

with open(output_file, 'w') as file:
    file.write('business_id_1, business_id_2, similarity\n')
    for row in final_pairs.collect():
        file.write(f'{row[0]},{row[1]},{row[2]}\n')

end = time.time()

print('total time:', end - start)