import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision import datasets
import torchvision.models as models

import pytorch_lightning as pl


# from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(nn.Linear(28 * 28, 64),
                                nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.l1(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 28 * 28))

    def forward(self, x):
        return self.l1(x)


class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.example_input_array = torch.Tensor(32, 1, 28, 28)
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        val_loss = F.mse_loss(x_hat, x)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = F.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# cli = LightningCLI(LitAutoEncoder)

# Load data sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(root="MNIST", download=True,
                           train=True, transform=transform)
test_set = datasets.MNIST(root="MNIST", download=True,
                          train=False, transform=transform)

# use 20% of training data for validation
train_set_size = int(len(train_set) * 0.8)
valid_set_size = len(train_set) - train_set_size

# split the train set into two
seed = torch.Generator().manual_seed(42)
train_set, valid_set = data.random_split(
    train_set, [train_set_size, valid_set_size], generator=seed)

train_loader = DataLoader(train_set)
valid_loader = DataLoader(valid_set)

# model
autoencoder = LitAutoEncoder(Encoder(), Decoder())

# train model
# trainer = pl.Trainer(fast_dev_run=7)
# trainer = pl.Trainer(limit_train_batches=0.1, limit_val_batches=0.01)
trainer = pl.Trainer(callbacks=[ModelSummary(max_depth=-1)])
trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=valid_loader)

# initialize the Trainer
# trainer = pl.Trainer()

# test the model
# trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))
