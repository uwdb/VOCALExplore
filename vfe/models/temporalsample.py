import math
import numpy as np

from .abstract import AbstractStrategy

class TemporalSampleStrategy(AbstractStrategy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = np.random.RandomState()

    @classmethod
    def name(cls):
        return 'temporalsample'

    def process_it(self, X_train, y_train, X_test, y_test, desired_count, stratify_train=None, stratify_test=None):
        for budget in self.budgets:
            num_done = self.num_done(budget=budget)
            print(f'num_done={num_done}')
            if num_done >= desired_count:
                continue
            sampled = self.sample(budget, X_train, y_train, stratify_train)
            if sampled is None:
                continue
            X_t, y_t = sampled
            model_results = self.train_and_evaluate_pytorch_model(X_t, y_t, X_test, y_test, it=num_done, model_name_suffix=f'_b{budget}', stratify_test=None)
            self.save_results([{
                **model_results,
                'budget': budget,
            }])

    def sample(self, budget, X, y, stratify):
        # budget is between 0 and 1.
        size = math.ceil(budget * len(X))
        prefix_fn = self.training_info.dataset.get_stratify_split_fn(self.name())
        nstratify_classes = len(set(map(prefix_fn, stratify)))

        # Sort by stratify.
        sort_order = np.argsort(stratify)

        group_size = math.ceil(size / nstratify_classes)
        past_prefix = ''
        first_instance_of_prefix = []
        for i in range(len(sort_order)):
            idx = sort_order[i]
            prefix = prefix_fn(stratify[idx])
            found_new_prefix = prefix != past_prefix
            if not found_new_prefix:
                continue
            past_prefix = prefix
            first_instance_of_prefix.append(i)

        train_idxs = []
        nprefixes = len(first_instance_of_prefix)
        for i, pidx in enumerate(first_instance_of_prefix):
            start_idx = pidx
            if i == nprefixes - 1:
                end_idx = min(start_idx + group_size, nprefixes)
            else:
                end_idx = min(start_idx + group_size, first_instance_of_prefix[i+1])

            train_idxs.extend(sort_order[start_idx:end_idx])

        return X[train_idxs], y[train_idxs]
