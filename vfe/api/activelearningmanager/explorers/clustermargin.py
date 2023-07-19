from collections import Counter, defaultdict
import fastcluster
import logging
import numpy as np
import random
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import cdist
from typing import Iterable, List

from vfe import core
from vfe.api.storagemanager import ClipInfo
from vfe.api.featuremanager.abstract import AbstractFeatureManager
from vfe.api.modelmanager.abstract import AbstractModelManager, PredictionSet
from vfe.api.videomanager.abstract import AbstractVideoManager
from vfe.api.activelearningmanager.abstractexplorer import AbstractExplorer
from vfe.api.activelearningmanager.explorers import RandomExplorer

from vfe.api.scheduler import UserPriority

logger = logging.getLogger(__name__)

# Set k_m = 10*k_t
# Set \epsilon such that the average cluster size of C is at least 10, so all clusters can be exhausted in round-robin sampling.
# hac: average linkage
# distancethreshold: binary search until we find a good value?

class ClusterMarginExplorer(AbstractExplorer):
    """Cluster-Margin sampling.

    Ref: https://arxiv.org/pdf/2107.14263.pdf
    """
    def __init__(self, rng=None, missing_vids_X=-1, instances_multiplier=10, random_steps=50, **kwargs):
        self.rng = np.random.default_rng(rng)
        self.missing_vids_X = missing_vids_X
        self.instances_multiplier = instances_multiplier

        self.random_steps = random_steps
        self.random = RandomExplorer(rng=rng, **kwargs)

        self.shown_fids = defaultdict(set)

        self.hac = {}
        self.hac_fids = defaultdict(set)

        logger.info(f'ClusterMarginExplorer(instances_multiplier={self.instances_multiplier})')

    def explore(self, feature_names: List[str], featuremanager: AbstractFeatureManager, modelmanager: AbstractModelManager, videomanager: AbstractVideoManager, k, t, label=None, vids=None, step=None) -> Iterable[ClipInfo]:
        logger.debug(f'ClusterMargin explore (features {feature_names}) over unlabeled vids with features')

        feature_name_str = core.typecheck.ensure_str(feature_names)

        if step * k < self.random_steps:
            logger.info('Falling back to random sampling for early step')
            return self.random.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids, step=step)

        # First check that we have more than one label detected.
        label_counts = modelmanager.get_label_counts(feature_names)
        if len(label_counts) < 2:
            logger.info('Falling back to random sampling because not enough classes')
            return self.random.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids, step=step)

        # Make sure there are enough features extracted.
        vids_with_features, vids_without_features = featuremanager.get_extracted_features_info(feature_names)
        labeled_vids = set(modelmanager.get_vids_with_labels())
        unlabeled_vids_with_features = set(vids_with_features) - labeled_vids
        logger.debug(f'{len(unlabeled_vids_with_features)} unlabeled vids with features; {len(vids_without_features)} vids without features')
        desired_vids = max(k, self.missing_vids_X)
        if len(unlabeled_vids_with_features) < desired_vids:
            candidate_vids = set(vids_without_features) - labeled_vids
            if candidate_vids:
                vids_to_extract = self.rng.choice(np.array([*candidate_vids]), size=min(desired_vids, len(candidate_vids)), replace=False)
                logger.info(f'Before doing cluster margin, extracting features from {len(vids_to_extract)} new vids')
                featuremanager.get_features(feature_names, vids_to_extract, priority=UserPriority.priority)

        # Get model predictions over all extracted features that aren't already labeled.
        # Make sure we use a model that can predict the specified label.
        # If vids is None, we'll get predictions over all stored features.
        try:
            y_pred_probs, model_labels, features = modelmanager.get_predictions(vids=vids, start=None, end=None, feature_names=feature_names, ignore_labeled=False, as_predictionset=False)
        except:
            logger.info('Falling back to random sampling because no predictions')
            # Random sample because we don't have enough to make predictions yet.
            return self.random.explore(feature_names, featuremanager, modelmanager, videomanager, k, t, label=label, vids=vids, step=step)

        # Cluster.
        current_fids = set(features['fid'].to_numpy())
        if feature_name_str not in self.hac or current_fids != self.hac_fids[feature_name_str]:
            hac = self.do_cluster(np.vstack(features['feature'].to_numpy()), avg_size=self.instances_multiplier * 5)
            self.hac[feature_name_str] = hac
            self.hac_fids[feature_name_str] = current_fids

        # Sort predictions by probability so we can get the difference between the first and second most likely.
        # This is slightly inefficient because we don't need to get predictions on labeled data, but when there are a small number of labels this shouldn't add too much cost.
        labeled_idx = np.where(features['labels'].to_numpy() != 'none')[0]
        sorted, indices = y_pred_probs.sort(axis=1)
        max_probs = sorted[:, -1]
        second_max_probs = sorted[:, -2]
        uncertainty_estimates = 1 + second_max_probs - max_probs
        argsort = np.argsort(-uncertainty_estimates)
        # Filter argsort based on the values, which represent indices.
        # Only keep values where the index is not one of the labeled ones.
        not_labeled = ~np.isin(argsort, labeled_idx)
        shown_fid_idxs = np.where(np.isin(features['fid'].to_numpy(), np.array(list(self.shown_fids[feature_name_str]))))[0]
        not_shown = ~np.isin(argsort, shown_fid_idxs)
        query_idx = argsort[not_shown & not_labeled][: int(self.instances_multiplier * k)]
        cluster_ids = self.hac[feature_name_str][query_idx]

        selected_idx = self.choose_cm_samples(
            cluster_ids, query_idx, k
        )
        sampled_clips = features.select(['fid', 'vid', 'start_time', 'end_time']).take(selected_idx).to_pylist()
        for c in sampled_clips:
            self.shown_fids[feature_name_str].add(c['fid'])
            logger.debug(f'Adding clip ({c["vid"]}, {c["start_time"]:.2f}-{c["end_time"]:.2f})')
        return [
            ClipInfo(vid=c['vid'], vstart=None, start_time=c['start_time'], end_time=c['end_time'])
            for c in sampled_clips
        ]

    def choose_cm_samples(self, cluster_ids, query_idx, k):
        # cluster_ids is numpy array. query_idx is a tensor.
        # Only get counts for the clusters that we are picking from.
        cluster_counts = Counter(cluster_ids)

        # query_idx are the indexes into the original retrieved data.
        cluster_to_query_idx = {
            cluster_id: query_idx[np.where(cluster_ids == cluster_id)[0]]
            for cluster_id in list(cluster_counts.keys())
        }

        sorted_cluster = [
            cid for ccount, cid in sorted([(v, k) for k, v in cluster_counts.items()])
        ]

        unsaturated_clusters = set(cluster_counts.keys())
        curr_idx = -1
        selected_query_idx = []
        logger.info(f'CM sampling with round-robin over {len(cluster_counts)} clusters')
        while unsaturated_clusters and len(selected_query_idx) < k:
            curr_idx = (curr_idx + 1) % len(sorted_cluster)
            curr_cluster = sorted_cluster[curr_idx]
            logger.debug(f'Trying cluster {curr_cluster}')
            if curr_cluster not in unsaturated_clusters:
                logger.debug(f'Cluster {curr_cluster} is saturated; skipping.')
                continue

            # Randomly select an unlabeled example from this cluster.
            query_idxs_in_cluster = cluster_to_query_idx[curr_cluster]
            if not len(query_idxs_in_cluster):
                unsaturated_clusters.remove(curr_cluster)
                logger.debug(f'Cluster {curr_cluster} has no unlabeled; skipping')
                continue

            logger.debug(f'Cluster {curr_cluster} has {len(query_idxs_in_cluster)} unlabeled')
            sample_idx = self.rng.choice(len(query_idxs_in_cluster), size=1)[0]
            selected_query_idx.append(query_idxs_in_cluster[sample_idx].item())
            # Mark this sample_idx as already labeled so that we don't pick it again this iteration.
            cluster_to_query_idx[curr_cluster] = np.delete(
                cluster_to_query_idx[curr_cluster], sample_idx
            )

        return np.array(selected_query_idx)


    # TODO: I don't think this is quite right.
    # The algorithm uses uncertainty sampling to select clusters, but then samples at random from the associated clusters.
    def __choose_cm_samples(cluster_labels, query_idx, k, random_query):
        """Inspired by https://github.com/AIRI-Institute/al_toolbox/blob/dev/acleto/al4nlp/query_strategies/cluster_margin_sampling.py"""
        cluster_sizes = Counter(cluster_labels)
        new_query_idx = []
        samples_idx = []
        # split all query_idx to array by cluster label
        query_idx_by_clusters = {
            idx: list(query_idx[np.where(cluster_labels == idx)])
            for idx in list(cluster_sizes.keys())
        }

        sorted_cluster = [
            el[1] for el in sorted([(v, k) for k, v in cluster_sizes.items()])
        ]

        curr_idx = 0
        while sum(cluster_sizes.values()) > 0 and len(new_query_idx) < k:
            # Sample data from each cluster
            # Sorted cluster: array with cluster numbers in ascending order by size
            # cluster_labels: dict with cluster sizes as values
            curr_cluster = sorted_cluster[curr_idx]
            if cluster_sizes[curr_cluster] == 0:
                curr_idx = (curr_idx + 1) % len(sorted_cluster)
                continue
            # randomly sample from data with curr_cluster labels
            if random_query:
                sample_idx = random.choice(
                    np.arange(len(query_idx_by_clusters[curr_cluster]))
                )
            else:
                sample_idx = 0
            samples_idx.append(sample_idx)
            new_query_idx.append(query_idx_by_clusters[curr_cluster][sample_idx])
            # remove this sample from data.
            query_idx_by_clusters[curr_cluster] = np.delete(
                query_idx_by_clusters[curr_cluster], sample_idx
            )
            # Subtract one from this cluster size.
            cluster_sizes[curr_cluster] -= 1
            curr_idx = (curr_idx + 1) % len(sorted_cluster)
        return np.array(new_query_idx)

    def do_cluster(self, X, avg_size=10):
        avg_cluster_size = avg_size + 1

        t = len(X)
        nclusters = t // avg_size
        logger.info(f't: {t}; nclusters: {nclusters}')

        approximate_threshold = 50000
        if t > approximate_threshold:
            # Threshold where linkage takes ~1 min.
            # Problematic when 20000 ~ number of clusters because each cluster has just 1-2 points.
            logger.info('Doing approximate clustering to save time.')

            sample_idxs = self.rng.choice(t, size=approximate_threshold, replace=False)
            sample_X = X[sample_idxs].astype(np.float16)
            linkage = fastcluster.linkage(sample_X, method='average', metric='euclidean', preserve_input=True)
            clusters = fcluster(linkage, t=nclusters, criterion='maxclust')
            # compute centroid for each cluster.
            centroids = np.array([
                np.average(sample_X[np.where(clusters == cluster)], axis=0)
                for cluster in np.unique(clusters)
            ]).astype(np.float16)

            # Then assign all feature vectors to closest cluster.
            hac = -1 * np.ones(t)
            batch_size = 10000
            for start_idx in range(0, t, batch_size):
                stop_idx = min(start_idx + batch_size, t)
                logger.debug(f'Computing distances for ({start_idx}, {stop_idx})')
                distances = cdist(centroids, X[start_idx:stop_idx].astype(np.float16), metric='euclidean')
                cluster_ids = np.argmin(distances, axis=0)
                hac[start_idx:stop_idx] = cluster_ids
        else:
            linkage = fastcluster.linkage(X, method='average', metric='euclidean', preserve_input=False)
            del X
            hac = fcluster(linkage, t=nclusters, criterion='maxclust')

        cluster_counts = Counter(hac)
        avg_cluster_size = np.mean(list(cluster_counts.values()))

        logger.debug(f'final -- avg_cluster_size: {avg_cluster_size}')
        return hac.astype(int)

def main():
    import os
    from vfe.api.storagemanager.basic import BasicStorageManager as StorageManager
    from vfe.api.featuremanager.basic import BasicFeatureManager

    core.logging.configure_logger()

    db_dir = '/gscratch/balazinska/mdaum/video-features-exploration/service/storage/deer/oracle/'
    features_dir = os.path.join(db_dir, 'features')
    sm = StorageManager(db_dir=db_dir, features_dir=features_dir, models_dir=None)
    fm = BasicFeatureManager(sm)
    feats = fm.get_features('r3d_18_ap_mean_stride32_flatten', vids=None).to_table()
    X = np.vstack(feats['feature'].to_numpy())

    cm = ClusterMarginExplorer()
    hac = cm.do_cluster(X, avg_size=10)
    print('here')


if __name__ == '__main__':
    main()


# With maxclust:
# deer does well with euclidean, avg size is 10