import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils import data

from . import fullysupervised, randomsample, lancet, temporalsample, labelreader, fuzzylabelreader

def get_strategy(strategy, *args, **kwargs):
    for cls in [
        fullysupervised.FullySupervisedStrategy,
        randomsample.RandomSampleStrategy,
        lancet.LancetStrategy,
        temporalsample.TemporalSampleStrategy,
        labelreader.LabelReaderStrategy,
        fuzzylabelreader.DurationFromCenterFuzzyLabelStrategy,
        fuzzylabelreader.DurationAccurateFuzzyLabelStrategy,
        fuzzylabelreader.DurationGaussianFuzzyLabelStrategy,
    ]:
        if strategy == cls.name():
            return cls(*args, **kwargs)
    raise RuntimeError(f'Unrecognized strategy {strategy}')

def evaluate_model(model_type_cls, ckpt_path, X, return_probs=False):
    model = model_type_cls.load_from_checkpoint(ckpt_path)
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=False,
        logger=False,
    )
    pt_dataset = data.TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(np.ones((len(X), 1))),
    )
    y_pred_probs = torch.stack(trainer.predict(model, ckpt_path=None, dataloaders=data.DataLoader(pt_dataset, num_workers=0))).squeeze()
    if return_probs:
        return y_pred_probs
    return np.argmax(y_pred_probs, axis=1).numpy()
