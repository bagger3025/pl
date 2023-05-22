import random
import numpy as np
import os
import pandas as pd

SEED = 42

SLOPE, BIAS = 1.5, 100
DATA_LEN = 12000
RANGE_MIN, RANGE_MAX = -100, 100
FLUC = 5

CUR_FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILEPATH = os.path.join(CUR_FILEPATH, "..")
os.makedirs(DATA_FILEPATH, exist_ok=True)

random.seed(SEED)

def get_data() -> list[tuple[float, float]]:
    x = [random.random() * (RANGE_MAX - RANGE_MIN) + RANGE_MIN for _ in range(DATA_LEN)]

    return [(_x, _x * SLOPE + BIAS + (random.random() * 2 - 1) * FLUC) for _x in x]

df = pd.DataFrame(np.array(get_data()), columns=['x', 'y'])
df.to_csv(os.path.join(DATA_FILEPATH, "regression_data.csv"), index=False)

print(df.head())

# np.save(os.path.join(DATA_FILEPATH, "my_data"), np.array(get_data()))
