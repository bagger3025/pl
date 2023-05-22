# [0, 100), [0, 100): class 1
# [0, 100), [100, 200): class 2
# [100, 200), [0, 100) : class 3
# [100, 200), [100, 200): class 4

import random
import os
import pandas as pd

import numpy as np

SEED = 42
DATA_LEN = 100_000
CUR_FILEPATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILEPATH = os.path.join(CUR_FILEPATH, "..")
os.makedirs(DATA_FILEPATH, exist_ok=True)

random.seed(SEED)

def get_class(x, y):
    if x < 100 and y < 100:
        return 0
    elif x < 100 and y < 200:
        return 1
    elif x < 200 and y < 100:
        return 2
    else:
        return 3

data = [(random.random() * 200, random.random() * 200) for _ in range(DATA_LEN)]
data = [(x, y, get_class(x, y)) for x, y in data]
data = np.array(data, dtype=np.float32)

df = pd.DataFrame(data, columns=['x', 'y', 'class'])
print(df.head())
df.to_csv(os.path.join(DATA_FILEPATH, "classifier_data.csv"), index=False)