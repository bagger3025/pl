import pytorch_lightning as pl

from util.data import regressionDataModule
from util.model import regressionLitModel

dm = regressionDataModule()

model = regressionLitModel()

trainer = pl.Trainer()

trainer.fit(model, dm)
