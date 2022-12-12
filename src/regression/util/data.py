import os
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torch.utils import data
import numpy as np


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA_DIR = os.path.join(BASE_DIR, "data")
MY_DATA_FILE = os.path.join(DATA_DIR, "my_data.npy")


class regressionDataSet(Dataset):
    def __init__(self, data_dir: str = MY_DATA_FILE, stage: str = "train"):

        if stage == "train" or stage == "val":
            self.data = np.load(data_dir)[:10000]
        elif stage == "test":
            self.data = np.load(data_dir)[10000:11000]
        elif stage == "predict":
            self.data = np.load(data_dir)[11000:]
        else:
            raise ValueError(
                "stage should be one of [train, val, test, predict], but got", stage)
        self.data = self.data.astype(np.float32)

    def __getitem__(self, index: int):

        return np.array([self.data[index][0]]), np.array([self.data[index][1]])

    def __len__(self):
        return self.data.shape[0]


class regressionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = MY_DATA_FILE, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str) -> None:
        full_set = regressionDataSet(self.data_dir)
        self.train_set, self.val_set = data.random_split(full_set, [
                                                         9000, 1000])
        self.test_set = regressionDataSet(self.data_dir, "test")
        self.predict_set = regressionDataSet(self.data_dir, "test")

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.predict_set, batch_size=1, shuffle=False)
