import os
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import numpy as np
import pandas as pd


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC_DIR = os.path.join(BASE_DIR, "..")
DATA_DIR = os.path.join(SRC_DIR, "..", "data")
DATA_NAME = "classifier_data.csv"

class classifierDataSet(Dataset):
    def __init__(self, df, stage: str = "train"):
        self.data = df
        print(self.data.head())

        self.features = df.loc[:, ["x", "y"]]
        self.classes = df.loc[:, "class"]

        if stage not in ['train', 'val', 'test', 'predict']:
            raise ValueError(
                f"stage should be one of [train, val, test, predict], but got {stage}")

    def __getitem__(self, index: int):
        return self.features.iloc[index], self.classes.iloc[index]

    def __len__(self):
        return self.data.shape[0]

df = pd.read_csv(os.path.join(DATA_DIR, DATA_NAME))

cd = classifierDataSet(df)
print(cd[3])


class classifierDataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        full_set = classifierDataSet(self.data_dir)
        self.train_set, self.val_set = data.random_split(full_set, [
                                                         85_000, 5_000])
        self.test_set = classifierDataSet(self.data_dir, "test")
        self.predict_set = classifierDataSet(self.data_dir, "predict")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=1, shuffle=False)
