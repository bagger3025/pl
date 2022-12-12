import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

class classifierLitModel(pl.LightningModule):

    def __init__(self, batch_size=32, learning_rate=1e-3):
        super().__init__()
        self.example_input_array = torch.Tensor(batch_size, 2)
        self.model = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 4), nn.ReLU())
        self.save_hyperparameters("batch_size", "learning_rate")

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.cross_entropy(x_hat, y)
        self.log("val_loss", loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer