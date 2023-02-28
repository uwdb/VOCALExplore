import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# From https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
class MLP(pl.LightningModule):
    def __init__(self, in_features, out_features):
        super().__init__()
        if 2/3 * in_features > out_features:
            hidden_features = int(2/3 * in_features)
        else:
            hidden_features = in_features

        self.hidden = nn.Linear(in_features, hidden_features)
        self.nonlinear = nn.ReLU()
        self.l1 = nn.Linear(hidden_features, out_features)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        intermediate = self.nonlinear(self.hidden(x))
        return self.l1(intermediate)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        # self.log('train_loss', loss, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return self.softmax(y_hat)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
