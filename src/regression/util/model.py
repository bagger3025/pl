import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class regressionLitModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, batch_size=32):
        super().__init__()
        self.example_input_array = torch.Tensor(batch_size, 1)
        self.model = nn.Sequential(nn.Linear(1, 1))
        self.save_hyperparameters("learning_rate", "batch_size")

    def forward(self, x):
        x = self.model(x)
        return x

    def on_train_start(self) -> None:
        self.logger.log_hyperparams(
            self.hparams, {"hp/metric_1": 0, "hp/metric_2": 1})

    def training_step(self, batch):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y) / 2
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, y) / 2
        self.log("val_loss", loss)

    def training_epoch_end(self, outputs) -> None:
        self.log("hp/metric_1", self.hparams["learning_rate"])
        self.log(
            "hp/metric_2", torch.tensor(self.hparams["batch_size"], dtype=torch.float32))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
