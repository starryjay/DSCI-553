from pyspark import SparkContext
import os

os.environ['PYSPARK_PYTHON'] = '/Users/shobhanashreedhar/anaconda3/envs/datamining/bin/python3.8'
os.environ['PYSPARK_DRIVER_PYTHON'] = '/Users/shobhanashreedhar/anaconda3/envs/datamining/bin/python3.8'

sc = SparkContext('local[*]', 'wordCount')

input_file_path = '/Users/shobhanashreedhar/Downloads/pg64317.txt'
textRDD = sc.textFile(input_file_path)

counts = textRDD.flatMap(lambda line: line.split(' ')).map(lambda word:(word,1)).reduceByKey(lambda a, b:a+b).collect()

for each_word in counts: 
    print(each_word)
