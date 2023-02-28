import numpy as np
import pytorch_lightning as pl
import torch

from vfe.models import linearmodel

def train_model(X, y, nclasses, model=None, learning_rate=0.02):
    if model is None:
        model = linearmodel.LinearModel(X.shape[1], nclasses, learning_rate)
    trainer = pl.Trainer(max_epochs=100, devices=1, accelerator='cpu', enable_progress_bar=False, enable_checkpointing=False, enable_model_summary=False, logger=False)

    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    trainer.fit(model, train_dataloaders=torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True))
    return trainer, model

def predict_model_logits(trainer, model, X, logits=False):
    predict_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(np.ones(len(X))))
    predictions = trainer.predict(model, ckpt_path=None, dataloaders=torch.utils.data.DataLoader(predict_dataset))
    y_pred_probs = torch.stack(predictions).squeeze()
    if logits:
        return np.argmax(y_pred_probs, axis=1)
    else:
        return y_pred_probs
