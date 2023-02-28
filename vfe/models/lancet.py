import numpy as np
import os

from .abstract import AbstractStrategy

class LancetStrategy(AbstractStrategy):
    def __init__(self, *args, indices_dir=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices_dir = indices_dir
        self.nlabels_to_indices = self._preprocess_indices(indices_dir)

    @classmethod
    def name(cls):
        return 'lancet'

    def process_it(self, X_train, y_train, X_test, y_test, desired_count):
        for nlabels, indices in self.nlabels_to_indices.items():
            budget = float(f'{nlabels / len(X_train):0.2f}')
            num_done = self.num_done(budget=budget)
            if num_done >= desired_count:
                continue
            X_t = X_train[indices]
            y_t = y_train[indices]
            model_results = self.train_and_evaluate_pytorch_model(X_t, y_t, X_test, y_test, it=num_done)
            self.save_results([{
                **model_results,
                'budget': budget,
            }])

    def _preprocess_indices(self, indices_dir):
        nlabels_to_indices = {}
        for file in os.listdir(indices_dir):
            unique_indices = np.unique(np.loadtxt(os.path.join(indices_dir, file)).astype(int))
            nlabels_to_indices[len(unique_indices)] = unique_indices
        return nlabels_to_indices
