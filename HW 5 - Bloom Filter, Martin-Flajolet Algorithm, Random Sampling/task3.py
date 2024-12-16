from blackbox import BlackBox
import sys
import random
import time

input_file = sys.argv[1]
stream_size = int(sys.argv[2])
num_of_asks = int(sys.argv[3])
output_file = sys.argv[4]

random.seed(553)

def fixed_size_sampling(stream_size: int, num_of_asks: int) -> list:
    reservoir = []
    res_size = 100
    seen = 0
    stage = 0
    out = ["seqnum,0_id,20_id,40_id,60_id,80_id"]
    for _ in range(num_of_asks):
        bx = BlackBox()
        stream = bx.ask(input_file, stream_size)
        for user in stream:
            seen += 1
            if seen <= res_size:
                reservoir.append(user)
            else:
                if random.random() < (100 / seen):
                    replace = random.randint(0, res_size - 1)
                    reservoir[replace] = user
            if seen % 100 == 0:
                stage = seen
                out.append(str(stage) + "," + reservoir[0] + "," + reservoir[20] + "," + reservoir[40] + "," + reservoir[60] + "," + reservoir[80])
    return out

def write_fss(out: list) -> None:
    with open(output_file, "w") as f:
        for line in out:
            f.write(line + "\n")

if __name__ == "__main__":
    start = time.time()
    output = fixed_size_sampling(stream_size, num_of_asks)
    write_fss(output)
    end = time.time()
    print("Duration:", end - start)