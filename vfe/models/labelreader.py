import duckdb
import numpy as np
import os

from .abstractlabeler import DatasetInfo, Dataset
from .abstract import AbstractStrategy

class LabelReaderStrategy(AbstractStrategy):
    def __init__(self, *args, label_strategy=None, dataset_info: DatasetInfo = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self._name
        self.label_strategy = label_strategy
        self.dataset_info = dataset_info
        # Database information is from abstractlabeler.py
        self.con = duckdb.connect(os.path.join(self.outdir, 'label_progress.duckdb'), read_only=True)

    def _name(self):
        return self.__class__.name() + '_' + self.label_strategy

    @classmethod
    def name(cls):
        return 'lr'

    def get_labeled_fids(self, budget, experiment_id):
        query = """
            SELECT iteration, fids
            FROM label_progress
            WHERE budget=?
                AND experiment=?
                AND dataset=?
                AND lower(feature)=?
                AND split_type=?
                AND split_idx=?
                AND strategy=?
        """
        return self.con.execute(query, [budget, experiment_id, *self.dataset_info, self.label_strategy]).fetchnumpy()

    def get_all_experiments(self, budget):
        query = """
            SELECT DISTINCT experiment
            FROM label_progress
            WHERE budget=?
                AND dataset=?
                AND lower(feature)=?
                AND split_type=?
                AND split_idx=?
                AND strategy=?
        """
        return set(self.con.execute(query, [budget, *self.dataset_info, self.label_strategy]).fetchnumpy()['experiment'])

    def get_done_experiments(self, budget):
        if not os.path.exists(self.outfile):
            return set([])
        df = self.results_df
        filtered = df[
            (df.budget == budget)
            & (df.dataset == self.dataset_info.name)
            & (df.feature == self.dataset_info.feature)
            & (df.split_type == self.dataset_info.split_type)
            & (df.split_idx == self.dataset_info.split_idx)
            & (df.label_strategy == self.label_strategy)
        ]

        return set(filtered['experiment'])

    def process_dataset(self, split_datasets):
        train_dataset: Dataset = split_datasets['train']
        test_dataset: Dataset = split_datasets['test']

        for budget in self.budgets:
            all_experiments = self.get_all_experiments(budget)
            done_experiments = self.get_done_experiments(budget)

            for experiment_id in all_experiments - done_experiments:
                # experiment_id is a numpy type. Convert it to a native python type,
                # which is necessary at least when using it in a duckdb query.
                experiment_id = experiment_id.item()
                labeled_fids = self.get_labeled_fids(budget, experiment_id)
                # For now just handle a single batch of labels.
                assert len(labeled_fids['iteration']) == 1, f'Expected one batch of labels but found {len(labeled_fids["iteration"])}'
                labeled_idxs = np.where(np.isin(train_dataset.fids, labeled_fids['fids'][0]))
                X_train = train_dataset.X[labeled_idxs]
                y_train = train_dataset.y[labeled_idxs]
                eval_info = {
                    'test': (test_dataset.X, test_dataset.y),
                    'l+u': (train_dataset.X, train_dataset.y),
                }
                results = self.train_and_evaluate_pytorch_model_multieval(X_train, y_train, eval_info, it=experiment_id, model_name_suffix=f'_b{budget}_{self.dataset_info.split_type}{self.dataset_info.split_idx}')
                self.save_results([{
                    **results,
                    'budget': budget,
                    'experiment': experiment_id,
                    'split_type': self.dataset_info.split_type,
                    'split_idx': self.dataset_info.split_idx,
                    'label_strategy': self.label_strategy,
                }])
