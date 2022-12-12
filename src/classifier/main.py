import torch
torch.manual_seed(42)

import pytorch_lightning as pl
from pytorch_lightning import loggers

from util.data import classifierDataModule
from util.model import classifierLitModel

BATCH_SIZE = 64

dm = classifierDataModule(batch_size=BATCH_SIZE)

model = classifierLitModel(batch_size=BATCH_SIZE)

tb_logger = loggers.TensorBoardLogger(save_dir=".", default_hp_metric=False)
trainer = pl.Trainer(max_epochs=10, logger=tb_logger)

trainer.fit(model, dm)