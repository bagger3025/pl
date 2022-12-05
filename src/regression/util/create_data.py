import random
import numpy as np
import os

SEED = 42

SLOPE, BIAS = 1.5, 100
DATA_LEN = 12000
RANGE_MIN, RANGE_MAX = -100, 100
FLUC = 5

CUR_FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILEPATH = os.path.join(CUR_FILEPATH, "..", "data")
os.makedirs(DATA_FILEPATH, exist_ok=True)

random.seed(SEED)

def get_data() -> list[tuple[float, float]]:
    x = [random.random() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN for _ in range(DATA_LEN)]

    return [(_x, _x * SLOPE + BIAS + (random.random() * 2 - 1) * FLUC) for _x in x]

np.save(os.path.join(DATA_FILEPATH, "my_data"), np.array(get_data()))
