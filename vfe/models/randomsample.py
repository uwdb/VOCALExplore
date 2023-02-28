import math
import numpy as np
import pandas as pd
from sklearn import model_selection

from .abstract import AbstractStrategy

class RandomSampleStrategy(AbstractStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState()

    @classmethod
    def name(cls):
        return 'randomsample'

    def process_it(self, X_train, y_train, X_test, y_test, desired_count, stratify_train=None, stratify_test=None):
        for budget in self.budgets:
            num_done = self.num_done(budget=budget)
            if num_done >= desired_count:
                continue
            sampled = self.sample(budget, X_train, y_train, stratify_train)
            if sampled is None:
                continue
            X_t, y_t = sampled
            model_results = self.train_and_evaluate_pytorch_model(X_t, y_t, X_test, y_test, it=num_done, model_name_suffix=f'_b{budget}', stratify_test=stratify_test)
            self.save_results([{
                **model_results,
                'budget': budget,
            }])

    def sample(self, budget, X, y, stratify):
        # budget is between 0 and 1.
        sample_includes_all_classes = False
        nclasses = len(set(y))
        mincount = 2
        size = math.ceil(budget * len(X))
        # if nclasses * mincount >= size:
        #     return None
        while not sample_includes_all_classes:
            # Test size should be >= number of classes.
            stratify = stratify if stratify is not None and len(set(stratify)) <= size else None
            X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=budget, random_state=self.rng, stratify=stratify) #, stratify=y)
            # idxs = self.rng.choice(len(X), size=size, replace=False)
            # Check that we include all classes and that each class shows up at least twice for cross-validation purposes.
            sample_includes_all_classes = True # = (len(set(y_te)) == nclasses) and (pd.Series(y_te).groupby(y_te).count().min() >= mincount)
        return X_te, y_te
