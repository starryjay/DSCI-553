from blackbox import BlackBox
import sys
import random
import time
import binascii
import math

def myhashs(s: str) -> list:
    return [(random.randint(1, 69997) * int(binascii.hexlify(s.encode('utf8')), 16) + 
             random.randint(1, 69997)) % ((10 ** 9) + 7) % 69997 for _ in range(3)]

input_file = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file = sys.argv[4]
existing_positions = set()
true_fpr = (1 - math.exp((-485 * stream_size) / 69997)) ** 485
print("True FPR: ", true_fpr)

def bloom_filter(stream: list, global_bit_array: list, users: set):
    potential_fp = {}
    true_negatives = {}
    for user in stream:
        user_positions = myhashs(user)
        if all(global_bit_array[position] == 1 for position in user_positions) and user not in users:
            potential_fp[user] = user_positions
        elif any(global_bit_array[position] == 0 for position in user_positions):
            true_negatives[user] = user_positions
        for position in user_positions:
            global_bit_array[position] = 1
        users.add(user)
    return potential_fp, true_negatives, global_bit_array, users

def get_fpr(bx: BlackBox, global_bit_array: list) -> dict:
    fpr_dict = {}
    users = set()
    for ask in range(num_of_asks):
        stream = bx.ask(input_file, stream_size)
        potential_fp, true_negatives, global_bit_array, users = bloom_filter(stream, global_bit_array, users)
        fpr = len(potential_fp) / (len(potential_fp) + len(true_negatives))
        fpr_dict[ask] = fpr
    return fpr_dict

def write_fpr(num_of_asks: int, fpr_dict: dict, output_file: str) -> None:
    with open(output_file, 'w') as f:
        f.write('Time,FPR\n')
        for i in range(num_of_asks):
            f.write(f'{i},{fpr_dict[i]}\n')

def main() -> None:
    start = time.time()
    bx = BlackBox()
    global_bit_array = [0] * 69997
    fpr_dict = get_fpr(bx, global_bit_array)
    write_fpr(num_of_asks, fpr_dict, output_file)
    end = time.time()
    print("Duration: ", end - start)

if __name__ == '__main__':
    main()