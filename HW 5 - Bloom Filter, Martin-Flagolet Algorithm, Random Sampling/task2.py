# Flajolet-Martin Algorithm

from blackbox import BlackBox
import sys
import random
import time
import binascii
import statistics
import math

input_file = sys.argv[1]
stream_size = int(sys.argv[2]) # will be 300
num_of_asks = int(sys.argv[3]) # should be greater than 30
output_file = sys.argv[4]
a = random.sample(range(1, 69997), 245)
b = random.sample(range(1, 69997), 245)
p = 10 ** 9 + 7
n = 69997

def myhashs(s: str) -> list:
    x = int(binascii.hexlify(s.encode('utf8')), 16)
    return [(a_ * x + b_ % p) % n for _, a_, b_ in zip(range(245), a, b)]

def flajolet_martin(stream: list):
    # In task2, you will implement the Flajolet-Martin algorithm 
    # including the step of combining estimations from groups of hash functions
    # to estimate the number of unique users within a window in the data stream.
    l = 0
    r = 10
    estimations = []
    while r < len(stream):
        R = 0
        for i in range(l, r):
            user_positions = myhashs(stream[i])
            trailing_zeros = 0
            for position in user_positions:
                trailing_zeros = max(trailing_zeros, len(bin(position)) - len(bin(position).rstrip('0')))
            R += trailing_zeros
        l += 1
        r += 1
        estimations.append(R / 10)
    return statistics.median(estimations)

def get_estimations(bx: BlackBox) -> dict:
    estimations = {}
    for ask in range(num_of_asks):
        stream = bx.ask(input_file, stream_size)
        estimation = flajolet_martin(stream)
        estimations[ask] = round(2 ** estimation)
    return estimations

def write_estimations(num_of_asks: int, estimations: dict, stream: list, output_file: str) -> None:
    total_est = 0
    total_gt = 0
    with open(output_file, 'w') as f:
        f.write('Time,Ground Truth,Estimation\n')
        for i in range(num_of_asks):
            gt = len(set(stream))
            total_est += estimations[i]
            total_gt += gt
            f.write(f'{i},{gt},{estimations[i]}\n')
    print("Ratio of estimation to ground truth:", total_est / total_gt)

def main() -> None:
    start = time.time()
    bx = BlackBox()
    estimations = get_estimations(bx)
    stream = bx.ask(input_file, stream_size)
    write_estimations(num_of_asks, estimations, stream, output_file)
    end = time.time()
    print("Duration: ", end - start)

if __name__ == '__main__':
    main()
    

    

