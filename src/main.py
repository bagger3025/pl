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

from pytorch_lightning.callbacks.early_stopping import EarlyStopping


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
        self.encoder = encoder
        self.decoder = decoder
        self.save_hyperparameters()

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


class CIFAR10Classifier(pl.LightningModule):
    def __init__(self):
        # init the pretrained LightningModule
        self.feature_extractor = LitAutoEncoder.load_from_checkpoint(
            "some/path")
        self.feature_extractor.freeze()

        # the autoencoder outputs a 100-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(100, 10)

    def forward(self, x):
        representations = self.feature_extractor(x)
        x = self.classifier(representations)
        # ...


class ImagenetTransferLearning(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        # ...


def finetune_imagenet_transferlearning():
    def some_images_from_cifar10():
        return

    model = ImagenetTransferLearning()
    trainer = pl.Trainer()
    trainer.fit(model)

    model = ImagenetTransferLearning.load_from_checkpoint("some/path")
    model.freeze()

    x = some_images_from_cifar10()
    predictions = model(x)


class BertMNLIFinetuner(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # from huggingface
        self.bert = BertModel.from_pretrained(
            "bert-base-cased", output_attentions=True)
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn


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
trainer = pl.Trainer(max_epochs=2, callbacks=[
                     EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=autoencoder, train_dataloaders=train_loader,
            val_dataloaders=valid_loader)

# initialize the Trainer
trainer = pl.Trainer()

# test the model
trainer.test(model=autoencoder, dataloaders=DataLoader(test_set))
