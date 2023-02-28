from collections import defaultdict
from enum import Enum
import functools
import logging
import threading
from typing import Iterable, List, Callable

class RisingBandit:
    # Implements Algorithm 1 from the following paper:
    # https://www.semanticscholar.org/reader/38acf4fe9df3e4f5b8d5e3315beb8d8b24f6f607

    def __init__(self,
            candidate_feature_names: Iterable[str] = None,
            C = 1,
            T = 1000,
            eval_candidate: Callable[[List[str]], float] = None,
            consider_all = False,
            async_step = False,):
        assert len(candidate_feature_names), f'Error: must specify at least one candidate feature'
        self.candidate_feature_names = set(candidate_feature_names)
        self.C = C
        self.T = T
        self.t = 0
        self.pruned_t = 0
        # pruned_t -> event
        self._pruned_events = {}
        self._eval_candidate = eval_candidate
        self._consider_all = consider_all
        self._async_step = async_step

        # async_step data structures:
        self.eval_progress_lock = threading.Lock()
        # step -> number of evaluation tasks
        self.eval_step_to_count = {}

        # Map from feature_name -> list of values for each step
        self.y_k_lock = threading.Lock()
        self.y_k = defaultdict(list)
        self.w_k = defaultdict(list)
        self.u_k = defaultdict(list)
        self.l_k = defaultdict(list)

        self.logger = logging.getLogger(__name__)

    def candidates(self) -> List[str]:
        with self.y_k_lock:
            return sorted(list(self.candidate_feature_names))

    @property
    def _t_index(self):
        # -1 because step skips the first iteration.
        # -1 again because the lists are 0-indexed, while steps are 1-indexed.
        return self.pruned_t - 2

    def _update_y(self, value, event=None, feature_name=None, expected_len=None):
        with self.y_k_lock:
            if expected_len and len(self.y_k[feature_name]) != expected_len:
                raise RuntimeError(f'Unexpected y_k[{feature_name}]: {len(self.y_k[feature_name])} != {expected_len}')
            self.y_k[feature_name].append(value)

        if event:
            self.logger.debug(f'Updating y for {feature_name}')
            event.set()

    def step(self, alm):
        self.t += 1

        if self.t == 1:
            # The first time around there isn't anything to evaluate because there
            # are not labels yet.
            # Increment pruned_t to keep it aligned with t since we aren't calling _prune.
            self.pruned_t += 1
            return

        self.logger.info(f'Rising bandit step {self.t}')

        with self.y_k_lock:
            self._pruned_events[self.t] = threading.Event()

        # Do feature evaluation in parallel if the scheduler allows.
        # Use a consistent order so that feature extraction can be done in the same order.
        if self._async_step:
            # Wait on the previous step's pruned event.
            # Once all of the features have updated y, then do prune.
            def wrapped_callback(value, feature_name=None, t=None):
                # Pass in the expected length; currently we don't have any guardrails to ensure that
                # we call update_y for timesteps in order. Live with it for now because it seems like an
                # unlikely scenario. If the exception is ever raised, deal with it then.
                # Before y_k is updated, we expect its length to be t-2 (-1 because for this step,
                #  -1 because we skip the first step)
                self._update_y(value, feature_name=feature_name, expected_len=t - 2)

                do_prune = False
                with self.eval_progress_lock:
                    self.eval_step_to_count[t] -= 1

                    if self.eval_step_to_count[t] == 0:
                        self.eval_step_to_count.pop(t)
                        do_prune = True

                if do_prune:
                    with self.y_k_lock:
                        event = self._pruned_events.get(t - 1)
                    if event:
                        event.wait()
                    self._prune(t)

            candidates = self.candidates()
            with self.eval_progress_lock:
                self.eval_step_to_count[self.t] = len(candidates)
            for feature_name in candidates:
                self._eval_candidate([feature_name], callback=functools.partial(wrapped_callback, feature_name=feature_name, t=self.t))
        else:
            events = []
            for feature_name in self.candidates():
                event = threading.Event()
                self._eval_candidate([feature_name], callback=functools.partial(self._update_y, event=event, feature_name=feature_name))
                events.append(event)
            for event in events:
                event.wait()

            # Prune needs to be for this t.
            # Prune needs to wait if prune for previous t hasn't been called yet.
            self._prune(self.t)

    def _prune(self, t):
        # self.candidates() gets the lock, so do this before this function gets the lock.
        candidates = self.candidates()
        with self.y_k_lock:
            self.pruned_t += 1

            if t != self.pruned_t:
                # This specific call to _prune is too early; we have to process earlier steps.
                self.logger.debug(f'_prune called with t={t}; currently at step {self.pruned_t}')
                raise ValueError()

            # Remove the event for the previous step since at this point we know the next step
            # is running, so nothing should be waiting on it.
            # We don't have an event for the first step, so skip that one.
            if self.pruned_t > 2:
                prev_event = self._pruned_events.pop(self.pruned_t - 1)
                assert prev_event.is_set()

            for feature_name in candidates:
                # self.y_k[feature_name].append(self._eval_candidate([feature_name]))
                self.l_k[feature_name].append(self._compute_lb(feature_name))
                self.w_k[feature_name].append(self._compute_growth_rate(feature_name))

                y_k = self.y_k[feature_name][self._t_index]
                w_k = self.w_k[feature_name][self._t_index]
                l_k = self.l_k[feature_name][self._t_index]
                # Replace y_k with l_k to get smoothed value.
                duration = self.T - self.t
                if duration <= 0:
                    self.logger.warn(f'Timestep is {self.t} but max step is {self.T}; using max={self.t + 1}')
                    duration = 1
                self.u_k[feature_name].append(min(l_k + w_k * duration, 1))

            self.logger.debug(f'After step {self.t} but before pruning, candidates are {self.candidate_feature_names}')
            for candidate in self.candidate_feature_names:
                self.logger.debug(f'  {candidate}: y_k={self.y_k[candidate]}, w_k={self.w_k[candidate]}, u_k={self.u_k[candidate]}, l_k={self.l_k[candidate]}')
            for candidate in set(self.y_k.keys()) - self.candidate_feature_names:
                self.logger.debug(f'  {candidate}: y_k={self.y_k[candidate]}, w_k={self.w_k[candidate]}, u_k={self.u_k[candidate]}, l_k={self.l_k[candidate]}')

            # We can evaluate growth rate once we have at least two values.
            if not self._consider_all and self.t > 2:
                ordered_candidates = sorted(list(self.candidate_feature_names))
                n_candidates = len(ordered_candidates)
                for i in range(n_candidates):
                    feature_i = ordered_candidates[i]
                    for j in range(n_candidates):
                        if i == j:
                            continue
                        feature_j = ordered_candidates[j]
                        if feature_j not in self.candidate_feature_names:
                            # If we already removed feature_j because it's upper bound is lower
                            # than some other feature's lower bound, we don't have to check it again.
                            continue
                        if self.l_k[feature_i][self._t_index] >= self.u_k[feature_j][self._t_index]:
                            self.logger.info(f'Removing candidate {feature_j} at step {self.t}')
                            self.candidate_feature_names.remove(feature_j)

            # Set prune_t event so that the next call to prune knows to proceed.
            self._pruned_events[self.pruned_t].set()

            self.logger.debug(f'After step {self.t} and after pruning, candidates are {self.candidate_feature_names}')

    def _compute_lb(self, feature_name):
        return self.y_k[feature_name][self._t_index]

    def _compute_growth_rate(self, feature_name):
        if self._t_index - self.C < 0:
            # We don't have enough points to average yet.
            # It's dangerous to average over a smaller span if we don't have enough points
            # because if the growth rate is noisy and the first step or two is slightly negative,
            # the upper bound when multiplied by (T-t) will be negative.
            return 1e6
        c_used = self._t_index - self.C
        # Compute based off of l_k to used smoothed value.
        interpolation_value = self.l_k[feature_name][c_used]
        while self.l_k[feature_name][self._t_index] - interpolation_value < 0:
            c_used -= 1
            if c_used < 0:
                c_used = 0
                interpolation_value = 0
            else:
                interpolation_value = self.l_k[feature_name][c_used]

        if interpolation_value == 0 and self.l_k[feature_name][self._t_index] == 0:
            # If everything is 0, then we don't have enough information yet to make a judgement call.
            # This is expected to happen only if the evaluation function returns 0 for more early rounds than C.
            return 1e6

        growth_rate = (self.l_k[feature_name][self._t_index] - interpolation_value) / (self._t_index - c_used)
        # If even after C steps the growth rate is negative, there is still too much noise.
        return growth_rate if growth_rate >= 0 else 1e6

class MovingAverageRisingBandit(RisingBandit):
    def __init__(self, *args, window=-1, **kwargs):
        super().__init__(*args, **kwargs)
        assert window > 0, 'Smoothing window must be specified'
        self.window = window

    def _compute_lb(self, feature_name):
        y_k = self.y_k[feature_name]
        if len(y_k) == 1:
            return y_k[0]
        start_index = max(0, self._t_index - self.window)
        return sum(y_k[start_index:self._t_index]) / (self._t_index - start_index)

class ExponentialWeightedAverageRisingBandit(RisingBandit):
    def __init__(self, *args, window=-1, **kwargs):
        super().__init__(*args, **kwargs)
        assert window > 0, 'Smoothing window must be specified'
        self.window = window
        self.alpha = 2.0 / (self.window + 1)
        # Feature name -> moving average.
        self.ewm = {}

    def _compute_lb(self, feature_name):
        y_k = self.y_k[feature_name]
        if feature_name not in self.ewm:
            assert len(y_k) == 1, f'Expected 1 value, got {len(y_k)}'
            self.ewm[feature_name] = [y_k[0]]
        else:
            ewm = (1 - self.alpha) * self.ewm[feature_name][-1] + self.alpha * y_k[self._t_index]
            self.ewm[feature_name].append(ewm)
        return self.ewm[feature_name][-1]

class CostNormalizedRisingBandit(ExponentialWeightedAverageRisingBandit):
    # TODO: Do we also need to adjust upper bound?
    def __init__(self, *args, cost_dict=None, cost_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert cost_dict, 'Cost dict must be specified'
        assert cost_weight, 'Cost weight must be specified'
        self.cost_dict = cost_dict
        self.cost_weight = cost_weight # Cost weight: if feature is 2x more expensive, how much better does performance need to be?

        min_cost = min(self.cost_dict.values())
        self.normalized_cost = {
            feature: cost / min_cost
            for feature, cost in self.cost_dict.items()
        }

    def _compute_lb(self, feature_name):
        # This call populates self.ewm[feature_name].
        super()._compute_lb(feature_name)

        # Normalize by the cost of this feature.
        if self.normalized_cost[feature_name] > 1:
            base_perf = self.ewm[feature_name][-1]
            normalized_perf = max((1 - self.normalized_cost[feature_name] / 2 * self.cost_weight) * base_perf, 0)
            self.ewm[feature_name][-1] = normalized_perf
        return self.ewm[feature_name][-1]


class BanditTypes(Enum):
    BASIC = 'basic'
    MOVING = 'moving'
    EXP = 'exp'
    WEIGHTED = 'weighted'

    def get_bandit(self, *args, **kwargs):
        if self == BanditTypes.BASIC:
            return RisingBandit(*args, **kwargs)
        elif self == BanditTypes.MOVING:
            return MovingAverageRisingBandit(*args, **kwargs)
        elif self == BanditTypes.EXP:
            return ExponentialWeightedAverageRisingBandit(*args, **kwargs)
        elif self == BanditTypes.WEIGHTED:
            return CostNormalizedRisingBandit(*args, **kwargs)
