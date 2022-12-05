import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F


class regressionLitModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.example_input_array = torch.Tensor(32, 1)
        self.model = nn.Sequential(nn.Linear(1, 1))
        self.save_hyperparameters()

    def forward(self, x):
        if x.dim() == 1:
            x = torch.unsqueeze(x, -1)
        x = self.model(x)
        return x

    def training_step(self, batch):
        x, y = batch
        x = x.to(torch.float)
        y = y.to(torch.float)
        
        x_hat = self.forward(x)
        if y.dim() == 1:
            y = torch.unsqueeze(y, -1)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer