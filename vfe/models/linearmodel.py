import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

# From https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example
class LinearModel(pl.LightningModule):
    def __init__(self, in_features, out_features, learning_rate, feature_name='', epochs=None, batch_size=None, use_cosine_annealing=False):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = nn.Linear(in_features, out_features)
        self.softmax = nn.Softmax(dim=1)
        self.learning_rate = learning_rate
        self.use_cosine_annealing = use_cosine_annealing

        # self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        # self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=out_features)
        # self.train_f1macro = torchmetrics.F1Score(num_classes=out_features, average='macro')
        # self.test_f1macro = torchmetrics.F1Score(num_classes=out_features, average='macro')

        # Logging train_mAP significantly slows down the training process.
        # self.test_mAP = torchmetrics.AveragePrecision(num_classes=out_features, average='macro')

    def forward(self, x):
        return self.l1(x)

    def on_train_start(self):
        if self.logger is None:
            return
        self.logger.log_hyperparams(self.hparams, {
            'hp/train/train_loss_step': 1e5,
            'hp/train/train_loss_epoch': 1e5,
            # 'hp/train/train_acc_step': 0,
            # 'hp/train/train_acc_epoch': 0,
            # 'hp/train/train_f1macro_step': 0,
            # 'hp/train/train_f1macro_epoch': 0,
            'hp/test/test_loss': 1e5,
            'hp/test/test_acc': 0,
            # 'hp/test/test_f1macro': 0,
            # 'hp/test/test_mAP': 0,
            'hp/params/learning_rate': self.learning_rate,
        })

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, self._normalize_y(y))
        # Double check initial loss is close to random chance.
        self.log('hp/train/train_loss', loss, on_step=True, on_epoch=True)

        # These metrics don't make sense for probabilistic inputs.
        # self.train_accuracy(y_hat, y)
        # self.log('hp/train/train_acc_step', self.train_accuracy)

        # self.train_f1macro(y_hat, y)
        # self.log('hp/train/train_f1macro_step', self.train_f1macro)

        self.log('hp/params/learning_rate', self.learning_rate)

        return loss

    # def training_epoch_end(self, outs):
    #     self.log('hp/train/train_acc_epoch', self.train_accuracy)
    #     self.log('hp/train/train_f1macro_epoch', self.train_f1macro)

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return self.softmax(y_hat)

    @staticmethod
    def _normalize_y(y):
        # Rescale each row so that the sum of all labels equals 1.
        # This avoids giving samples with multiple labels uneven weight in the training process.
        # This creates a target vector as described in the following paper:
        #   https://openaccess.thecvf.com/content_ECCV_2018/papers/Dhruv_Mahajan_Exploring_the_Limits_ECCV_2018_paper.pdf
        return nn.functional.normalize(y.float(), p=1, dim=1)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # self.test_accuracy(y_hat, y)
        # self.test_f1macro(y_hat, y)
        # self.test_mAP(y_hat, y)
        self.log('hp/test/test_loss', F.cross_entropy(y_hat, self._normalize_y(y)), on_step=False, on_epoch=True)

    # def validation_epoch_end(self, validation_step_outputs):
        # self.log('hp/test/test_acc', self.test_accuracy)
        # self.log('hp/test/test_f1macro', self.test_f1macro)
        # self.log('hp/test/test_mAP', self.test_mAP)

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        if not self.use_cosine_annealing:
            return opt
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [scheduler]
