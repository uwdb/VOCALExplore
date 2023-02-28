from collections import namedtuple
import duckdb
import os
from typing import List

Dataset = namedtuple('Dataset', ['X', 'y', 'y_composite', 'start_times', 'end_times', 'fids', 'vids'])
DatasetInfo = namedtuple('DatasetInfo', ['name', 'feature', 'split_type', 'split_idx'])
LabelProgress = List[List[int]]

class AbstractLabeler:
    def __init__(self,
        *,
        outdir = None,
        budgets: List = None,
    ):
        self.outdir = outdir
        self.budgets = budgets if budgets is not None else [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9]
        self.con = self._get_dbcon()

    def label_progress(self, dataset: Dataset, train_size) -> LabelProgress:
        # Return [[fid,]].
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError

    def reset_experiment(self):
        pass

    def reset_dataset(self, dataset: Dataset):
        pass

    def process_dataset_it(self, dataset: Dataset, dataset_info: DatasetInfo, desired_count):
        self.reset_experiment()
        for budget in self.budgets:
            num_done = self.num_done(budget=budget, dataset_info=dataset_info)
            if num_done >= desired_count:
                continue
            label_progress = self.label_progress(dataset, budget)
            self.save_results(budget, num_done+1, label_progress, dataset_info)

    def label_dataset(self, dataset: Dataset, dataset_info: DatasetInfo, desired_count):
        self.reset_dataset(dataset)
        for _ in range(desired_count):
            self.process_dataset_it(dataset, dataset_info, desired_count)

    def num_done(self, budget, dataset_info):
        query = """
            SELECT max(experiment)
            FROM label_progress
            WHERE budget=?
                AND dataset=?
                AND feature=?
                AND split_type=?
                AND split_idx=?
                AND strategy=?
        """
        result = self.con.execute(query, [budget, *dataset_info, self.name()]).fetchall()[0][0]
        return 0 if result is None else result

    def _initialize_db(self, db_path):
        create_tables_stmt = """
            CREATE TABLE label_progress(
                dataset VARCHAR,
                feature VARCHAR,
                split_type VARCHAR,
                split_idx UINTEGER,
                strategy VARCHAR,
                budget DECIMAL,
                experiment UINTEGER,
                iteration UINTEGER,
                fids UINTEGER[]
            )
        """
        con = duckdb.connect(db_path)
        con.execute(create_tables_stmt)

    def _get_dbcon(self):
        db_path = os.path.join(self.outdir, 'label_progress.duckdb')
        if not os.path.exists(db_path):
            if not os.path.exists(self.outdir):
                os.makedirs(self.outdir)
            self._initialize_db(db_path)
        return duckdb.connect(db_path)

    def save_results(self, budget, num_done, label_progress: LabelProgress, dataset_info: DatasetInfo):
        insert_stmt = """
                INSERT INTO label_progress
                (dataset, feature, split_type, split_idx, strategy, budget, experiment, iteration, fids)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
        self.con.execute("BEGIN TRANSACTION")
        for i, iteration_fids in enumerate(label_progress):
            self.con.execute(insert_stmt, [*dataset_info, self.name(), budget, num_done, i, iteration_fids.tolist()])
        self.con.execute("COMMIT")
