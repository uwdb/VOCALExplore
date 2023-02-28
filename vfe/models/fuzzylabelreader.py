from collections import defaultdict
import duckdb
import numpy as np
import os
from scipy import stats
from sklearn import preprocessing

from .abstractlabeler import DatasetInfo, Dataset
from .abstract import AbstractStrategy, to_multilabel_gen

class AbstractFuzzyLabelReaderStrategy(AbstractStrategy):
    def __init__(self, *args, label_strategy=None, dataset_info: DatasetInfo = None, before_duration=15, after_duration=15, only_use_correctly_augmented=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = self._name
        self.label_strategy = label_strategy
        self.dataset_info = dataset_info

        # Augment training.
        self.before_duration = before_duration
        self.after_duration = after_duration
        self.only_use_correctly_augmented = only_use_correctly_augmented
        self.ignore_labels = ['none', 'neutral']
        self._initialize_augmented_quality_tables()

        # Database information is from abstractlabeler.py
        self.con = duckdb.connect(self.db_path, read_only=True)

    @property
    def db_path(self):
        return os.path.join(self.outdir, 'label_progress.duckdb')

    @property
    def tmp_db_path(self):
        return os.path.join(self.outdir, 'tmp_label_progress.duckdb')

    def _name_minus_label(self):
        return self.__class__.name() \
            + (f'-b{self.before_duration}-a{self.after_duration}' if self.before_duration != 15 and self.after_duration != 15 else '')\
            + ('-oc' if self.only_use_correctly_augmented else '')

    def _name(self):
       return self._name_minus_label() + '_' + self.label_strategy

    @classmethod
    def name(cls):
        raise NotImplementedError

    def label_fn(self, dataset: Dataset, center_idx, to_label_idx):
        raise NotImplementedError

    def get_labeled_fids(self, budget, experiment_id):
        query = """
            SELECT iteration, fids
            FROM label_progress
            WHERE budget=?
                AND experiment=?
                AND dataset=?
                AND lcase(feature)=?
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
                AND lcase(feature)=?
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

    def label_range_around_fid(self, fid_idx, dataset: Dataset, start_time_sort_order, y_train_arr, label_encoder):
            span_start_time = dataset.start_times[fid_idx] - np.timedelta64(self.before_duration, 's')
            span_end_time = dataset.start_times[fid_idx] + np.timedelta64(self.after_duration, 's')

            for i in start_time_sort_order:
                if dataset.start_times[i] < span_start_time:
                    continue
                if dataset.start_times[i] > span_end_time:
                    break
                if dataset.start_times[i] >= span_start_time and dataset.end_times[i] <= span_end_time:
                    labels, prob = self.label_fn(dataset, fid_idx, i)
                    labels = [l for l in labels.split('_') if l not in self.ignore_labels]
                    label_idxs = label_encoder.transform(labels)
                    for label_idx in label_idxs:
                        y_train_arr[i][label_idx] = max(prob, y_train_arr[i][label_idx])
            return y_train_arr

    def _augment_training(self, labeled_fids, dataset: Dataset, y_train_arr, label_encoder):
        sort_order = np.argsort(dataset.start_times)
        for fid in labeled_fids:
            # The dataset may not contain the specified fid if we filtered it out based on its label.
            fid_idx = np.where(dataset.fids == fid)[0] # Result looks like (array([val]),).
            if len(fid_idx) == 0:
                print(f"Skipping fid {fid} because it's not in the training dataset")
                continue
            fid_idx = fid_idx[0]
            self.label_range_around_fid(fid_idx, dataset, sort_order, y_train_arr, label_encoder)
        return y_train_arr

    def _get_labels(self, split_datasets):
        labels = set(split_datasets['train'].y) | set(split_datasets['test'].y)
        for label in labels.copy():
            for sub_label in label.split('_'):
                labels.add(sub_label)
            if '_' in label or label in self.ignore_labels:
                labels.remove(label)
        return labels

    def _initialize_augmented_quality_tables(self):
        try:
            write_con = duckdb.connect(self.tmp_db_path, read_only=False)
        except Exception as e:
            print(f'Failed to get write connection to initialize augmented quality tables. This is only a problem if --compute-delta was specified.')
            return

        write_con.execute("""
            CREATE TABLE IF NOT EXISTS augmented_aggregates(
                dataset VARCHAR,
                feature VARCHAR,
                split_type VARCHAR,
                split_idx UINTEGER,
                strategy VARCHAR,
                label_strategy VARCHAR,
                budget DECIMAL,
                experiment UINTEGER,
                n_labeled UINTEGER,
                n_augmented UINTEGER,
                n_correctly_augmented UINTEGER,
                n_incorrectly_augmented UINTEGER,
                avg_row_delta DECIMAL
            )
        """)

        write_con.execute("""
            CREATE TABLE IF NOT EXISTS augmented_classvals(
                dataset VARCHAR,
                feature VARCHAR,
                split_type VARCHAR,
                split_idx UINTEGER,
                strategy VARCHAR,
                label_strategy VARCHAR,
                budget DECIMAL,
                experiment UINTEGER,
                variable VARCHAR,
                class VARCHAR,
                value DECIMAL
            )
        """)

    def check_done(self, con, dataset_info: DatasetInfo, budget, experiment_id):
        ndone = con.execute("""
            SELECT COUNT(*)
            FROM augmented_aggregates
            WHERE dataset=?
                AND feature=?
                AND split_type=?
                AND split_idx=?
                AND strategy=?
                AND label_strategy=?
                AND budget=?
                AND experiment=?
        """, [*dataset_info, self._name_minus_label(), self.label_strategy, budget, experiment_id]).fetchall()[0][0]
        return ndone > 0

    def compare_augmented_vs_labeled(self, dataset_info: DatasetInfo, split_datasets, budget):
        # Skip flr-duration-exact because there aren't any incorrectly augmented points.
        if 'exact' in self.name():
            return

        write_con = duckdb.connect(self.tmp_db_path, read_only=False)
        train_dataset: Dataset = split_datasets['train']
        all_experiments = self.get_all_experiments(budget)
        for experiment_id in all_experiments:
            experiment_id = experiment_id.item()
            print('***', *dataset_info, budget, experiment_id)
            if self.check_done(write_con, dataset_info, budget, experiment_id):
                print('Skipping: already done')
                continue
            augmented_y_train, label_encoder = self.get_augmented_y_train(split_datasets, budget, experiment_id)
            labeled_fids = self.get_labeled_fids(budget, experiment_id)['fids'][0]
            augmented_idxs = np.where(np.any(augmented_y_train != 0, axis=1))
            labeled_idxs = np.where(np.isin(train_dataset.fids, labeled_fids))
            labels, to_multilabel = to_multilabel_gen(self._get_labels(split_datasets))
            assert np.all(labels == label_encoder.classes_)
            y_true_transformed = to_multilabel(train_dataset.y)

            # Delta in augmented:
            delta_augmented = y_true_transformed[augmented_idxs] - augmented_y_train[augmented_idxs]
            incorrectly_augmented_idxs = np.where(np.any(delta_augmented != 0, axis=1))
            n_incorrectly_augmented = len(incorrectly_augmented_idxs[0])
            if n_incorrectly_augmented == 0:
                continue
            n_correctly_augmented = len(delta_augmented) - n_incorrectly_augmented
            n_labeled = len(labeled_idxs[0])
            n_augmented = len(augmented_idxs[0])

            # Which class has the most average delta?
            incorrectly_augmented_deltas = delta_augmented[incorrectly_augmented_idxs]
            avg_delta_by_class = np.mean(incorrectly_augmented_deltas, axis=0)
            total_positive_delta_by_class = np.sum(np.where(incorrectly_augmented_deltas < 0, 0, incorrectly_augmented_deltas), axis=0)
            total_negative_delta_by_class = np.sum(np.where(incorrectly_augmented_deltas < 0, incorrectly_augmented_deltas, 0), axis=0)
            # What is the average total delta per sample?
            avg_delta_per_row = np.mean(np.sum(delta_augmented, axis=1))

            write_con.execute("BEGIN TRANSACTION")
            write_con.execute("""
                INSERT INTO augmented_aggregates
                (dataset, feature, split_type, split_idx, strategy, label_strategy, budget, experiment, n_labeled, n_augmented, n_correctly_augmented, n_incorrectly_augmented, avg_row_delta)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [*dataset_info, self._name_minus_label(), self.label_strategy, budget, experiment_id, n_labeled, n_augmented, n_correctly_augmented, n_incorrectly_augmented, avg_delta_per_row])
            def add_class_vals(variable, values):
                for label, value in zip(labels, values):
                    write_con.execute("""
                        INSERT INTO augmented_classvals
                        (dataset, feature, split_type, split_idx, strategy, label_strategy, budget, experiment, variable, class, value)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, [*dataset_info, self._name_minus_label(), self.label_strategy, budget, experiment_id, variable, label, value])
            add_class_vals('avg_delta', avg_delta_by_class)
            add_class_vals('total_positive_delta', total_positive_delta_by_class)
            add_class_vals('total_negative_delta', total_negative_delta_by_class)
            write_con.execute("COMMIT")

    def get_augmented_y_train(self, split_datasets, budget, experiment_id):
        train_dataset: Dataset = split_datasets['train']
        test_dataset: Dataset = split_datasets['test']
        labeled_fids = self.get_labeled_fids(budget, experiment_id)
        # For now just handle a single batch of labels.
        assert len(labeled_fids['iteration']) == 1, f'Expected one batch of labels but found {len(labeled_fids["iteration"])}'
        labeled_fids = labeled_fids['fids'][0]

        labels = self._get_labels(split_datasets)
        label_encoder = preprocessing.LabelEncoder().fit(list(labels))
        augmented_y_train = np.zeros((len(train_dataset.y), len(label_encoder.classes_)))

        augmented_y_train = self._augment_training(labeled_fids, train_dataset, augmented_y_train, label_encoder)
        return augmented_y_train, label_encoder

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

                augmented_y_train, label_encoder = self.get_augmented_y_train(split_datasets, budget, experiment_id)

                # Find rows where at least one label has non-zero probability.
                # These are rows that were either explicitly labeled or were augmented.
                labeled_idxs = np.where(np.any(augmented_y_train != 0, axis=1))

                if self.only_use_correctly_augmented:
                    labels, to_multilabel = to_multilabel_gen(self._get_labels(split_datasets))
                    assert np.all(labels == label_encoder.classes_)
                    y_true_transformed = to_multilabel(train_dataset.y)
                    correctly_augmented_idxs = np.where(
                        np.all(y_true_transformed == augmented_y_train, axis=1)
                        & (np.any(y_true_transformed > 0, axis=1))
                    )
                    labeled_idxs = correctly_augmented_idxs
                    # Double check that we're  only including rows with at least one of the non-ignored labels.
                    assert np.min(np.sum(y_true_transformed[labeled_idxs], axis=1)) > 0

                X_train = train_dataset.X[labeled_idxs]
                y_train = augmented_y_train[labeled_idxs]

                eval_info = {
                    'test': (test_dataset.X, test_dataset.y),
                    'l+u': (train_dataset.X, train_dataset.y),
                }
                results = self.train_and_evaluate_pytorch_model_multieval(
                    X_train,
                    y_train,
                    eval_info,
                    it=experiment_id,
                    model_name_suffix=f'_b{budget}_{self.dataset_info.split_type}{self.dataset_info.split_idx}',
                    label_encoder=label_encoder,
                )
                self.save_results([{
                    **results,
                    'budget': budget,
                    'experiment': experiment_id,
                    'split_type': self.dataset_info.split_type,
                    'split_idx': self.dataset_info.split_idx,
                    'label_strategy': self.label_strategy,
                }])

class DurationFromCenterFuzzyLabelStrategy(AbstractFuzzyLabelReaderStrategy):
    @classmethod
    def name(cls):
        return 'flr-duration-from-center'

    def label_fn(self, dataset: Dataset, center_idx, label_idx):
        return dataset.y[center_idx], 1

class DurationAccurateFuzzyLabelStrategy(AbstractFuzzyLabelReaderStrategy):
    @classmethod
    def name(cls):
        return 'flr-duration-exact'

    def label_fn(self, dataset: Dataset, center_idx, label_idx):
        return dataset.y[label_idx], 1

class DurationGaussianFuzzyLabelStrategy(AbstractFuzzyLabelReaderStrategy):
    def __init__(self, *args, scale=5, **kwargs):
        super().__init__(*args, **kwargs)
        rv = stats.norm(loc=0, scale=scale)
        self.get_prob = lambda d: rv.pdf(d) / rv.pdf(0)

    @classmethod
    def name(cls):
        return 'flr-duration-gaussian'

    def label_fn(self, dataset: Dataset, center_idx, label_idx):
        # Probability using scale=5 as distance from center increases:
        # ['1.000000', '0.980199', '0.923116', '0.835270', '0.726149', '0.606531', '0.486752', '0.375311', '0.278037', '0.197899', '0.135335', '0.088922', '0.056135', '0.034047', '0.019841']
        delta_seconds = abs(dataset.start_times[center_idx] - dataset.start_times[label_idx]) / np.timedelta64(1, 's')
        return dataset.y[center_idx], self.get_prob(delta_seconds)
