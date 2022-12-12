import pytorch_lightning as pl
from pytorch_lightning import loggers

from util.data import regressionDataModule
from util.model import regressionLitModel

BATCH_SIZE = 128

dm = regressionDataModule(batch_size=BATCH_SIZE)

model = regressionLitModel(batch_size=BATCH_SIZE)

tb_logger = loggers.TensorBoardLogger(save_dir=".", default_hp_metric=False)
trainer = pl.Trainer(max_epochs=20, logger=tb_logger)

trainer.fit(model, dm)
