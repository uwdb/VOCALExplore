from collections import defaultdict
import datetime
import glob
import numpy as np
import os
import pandas as pd
from sklearn import metrics
import time
from torch.utils.tensorboard import SummaryWriter

from .abstractlabeler import AbstractLabeler
from .model_utils import train_model, predict_model_logits

class AbstractCoordinator:
    def __init__(self, Xs, y, labeler: AbstractLabeler, budget=-1, log_name=None, log_every_round=5, base_cluster_size=20, output_dir=None, log_kws={}, save_replay=False, X_test=None, y_test=None):
        # Xs: {feature_name: X}
        # y: true labels
        # budget: number of rounds of human interaction
        # log_name: identifier for this experiment. Should include colordinator type, dataset, feature(s), and labeler type. It will be automatically extended to include the current date/time.
        # log_every_round: how often to update dataframe on disk
        # output_dir: where to save dataframe
        self.Xs = Xs
        self.y = y
        self.nclasses = len(set(y))
        self.budget = budget
        self.labeler = labeler
        dt = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_name = self.decorate_log_name(self.labeler, log_name)
        self.log_name = log_name + '_dt_' + dt
        self.writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard', log_name, dt))
        self.log_every_round = log_every_round
        self.output_dir = output_dir
        self.log_kws = log_kws
        self.save_replay = save_replay
        self.base_cluster_size = base_cluster_size
        self.X_test = X_test
        self.y_test = y_test
        self.test_accuracy_every_round = 10

        # Keep track of stats.
        self.num_shown_to_labeler = []
        self.num_propagated = []
        self.num_labeled = []
        self.num_correct = []
        self.f1_macro = []
        self.test_metrics = defaultdict(list)
        self.extra_args = defaultdict(list)
        self.y_pred = -1 * np.ones(len(y))

        # Keep track of per-round information for replaying.
        self.label_progress = self.y_pred.copy()
        self.cluster_progress = self.y_pred.copy()
        self.labeled_idx_progress = [np.array([-1])]
        self.hard_labels_progress = self.y_pred.copy()
        self.transformed_X_progress = None

    @property
    def round(self):
        return len(self.num_shown_to_labeler)

    def num_already_done(self):
        log_prefix = self.log_name.split('_dt_')[0]
        df_files = glob.glob(os.path.join(self.output_dir, f'{log_prefix}*.pkl'))
        if len(df_files) == 0:
            return 0

        df = pd.concat([pd.read_pickle(f) for f in df_files])
        if self.budget == -1:
            # Count the number of cases where we labeled everything.
            num_done = len(np.where(df.groupby('logname')['num_labeled'].max() == len(self.y))[0])
        else:
            # Count the number of cases where we reached the budget.
            num_done = len(np.where(df.groupby('logname')['round'].max() == self.budget - 1)[0])
        return num_done

    def done_labeling(self):
        if self.budget == -1:
            return len(np.where(self.y_pred == -1)[0]) == 0
        else:
            return self.round >= self.budget

    def _get_model_accuracy(self):
        trainer, model = train_model(self.X[self.y_pred != -1], self.y[self.y_pred != -1], self.nclasses)
        y_pred = predict_model_logits(trainer, model, self.X_test, logits=True)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        f1_macro = metrics.f1_score(self.y_test, y_pred, average='macro')
        return accuracy, f1_macro

    def _save_results(self):
        output_file = os.path.join(self.output_dir, self.log_name + '.pkl')
        df = pd.DataFrame.from_dict({
            'round': list(range(self.round)),
            'num_shown_to_labeler': self.num_shown_to_labeler,
            'num_propagated': self.num_propagated,
            'num_labeled': self.num_labeled,
            'num_correct': self.num_correct,
            'fraction_correct': [nc / len(self.y) for nc in self.num_correct],
            'f1_macro': self.f1_macro,
            **self.test_metrics,
            **self.extra_args,
            **{
                k: [v] * self.round
                for k, v in self.log_kws.items()
            },
            'coordinator': self.name(),
            'labeler': self.labeler.name(),
            'logname': self.log_name,
            'base_cluster_size': self.base_cluster_size,
        })
        df.to_pickle(output_file)

        # Save replay arrays to output file.
        if self.save_replay:
            replay_arrays = {
                'label_progress': self.label_progress,
                'label_idx_progress': np.array(self.labeled_idx_progress, dtype=object),
                'cluster_progress': self.cluster_progress,
                'hard_labels_progress': self.hard_labels_progress,
                'transformed_X_progress': self.transformed_X_progress,
            }
            progress_dir = os.path.join(self.output_dir, 'progress')
            if not os.path.exists(progress_dir):
                os.mkdir(progress_dir)
            for name, arr in replay_arrays.items():
                if arr is None:
                    continue
                output_file = os.path.join(progress_dir, self.log_name + '_' + name + '.npy')
                np.save(output_file, arr)

    def _update_stats(self, new_y, new_y_idxs, idxs_shown_to_labeler, num_propagated, cluster_assignments=None, hard_labels=None, transformedX=None, **kwargs):
        self.num_shown_to_labeler.append(len(idxs_shown_to_labeler))
        self.num_propagated.append(num_propagated)
        self.y_pred[new_y_idxs] = new_y

        labeled_y_pred_idxs = np.where(self.y_pred != -1)[0]
        self.num_labeled.append(len(labeled_y_pred_idxs))
        self.num_correct.append(len(np.where(self.y_pred[labeled_y_pred_idxs] == self.y[labeled_y_pred_idxs])[0]))
        self.f1_macro.append(metrics.f1_score(self.y, self.y_pred, average='macro'))
        for k, v in kwargs.items():
            self.extra_args[k].append(v)

        if self.round % self.test_accuracy_every_round == 0 or self.done_labeling():
            test_accuracy, test_f1_macro = self._get_model_accuracy()
        else:
            test_accuracy, test_f1_macro = -1, -1
        self.test_metrics['test_accuracy'].append(test_accuracy)
        self.test_metrics['test_f1_macro'].append(test_f1_macro)

        # Writer.
        n_iter = self.round
        self.writer.add_scalar('num_shown_to_labeler', self.num_shown_to_labeler[-1], n_iter)
        self.writer.add_scalar('num_propagated', self.num_propagated[-1], n_iter)
        self.writer.add_scalar('num_labeled', self.num_labeled[-1], n_iter)
        self.writer.add_scalar('num_correct', self.num_correct[-1], n_iter)
        self.writer.add_scalar('fraction_correct', self.num_correct[-1] / len(self.y), n_iter)
        for k, v in kwargs.items():
            self.writer.add_scalar(k, v, n_iter)

        # Replay.
        self.label_progress = np.vstack([self.label_progress, self.y_pred])
        self.labeled_idx_progress.append(idxs_shown_to_labeler)
        self.cluster_progress = np.vstack([self.cluster_progress, cluster_assignments if cluster_assignments is not None else -1 * np.ones_like(self.y_pred)])
        self.hard_labels_progress = np.vstack([self.hard_labels_progress, hard_labels if hard_labels is not None else -1 * np.ones_like(self.y_pred)])
        if transformedX is not None:
            self.transformed_X_progress = np.expand_dims(transformedX, axis=0) if self.transformed_X_progress is None else np.vstack([self.transformed_X_progress, np.expand_dims(transformedX, axis=0)])

        # Save results.
        if self.round % self.log_every_round == 0 or self.done_labeling():
            self._save_results()

    def _interaction_round(self):
        # Return a dict with the args to _update_stats.
        raise NotImplementedError

    def interaction_round(self):
        start_round = time.perf_counter()
        start_round_p = time.process_time()

        rv_kwargs = self._interaction_round()

        round_time = time.perf_counter() - start_round
        round_time_p = time.process_time() - start_round_p

        self._update_stats(
            **rv_kwargs,
            # Extra args.
            round_time=round_time,
            round_time_p=round_time_p
        )

    @classmethod
    def name(cls):
        raise NotImplementedError

    @classmethod
    def decorate_log_name(cls, labeler: AbstractLabeler, suffix):
        return f'{cls.name()}_{labeler.name()}_{suffix}'
