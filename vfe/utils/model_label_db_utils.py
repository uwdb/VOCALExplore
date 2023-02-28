import numpy as np
import pyarrow as pa

from vfe.models.abstractlabeler import Dataset
from vfe import features

def get_extractor(model, **kwargs):
    if model == 'resnet18':
        return features.pretrained.models.Resnet18Extractor.create(**kwargs)
    elif model == 'inception_v3':
        return features.pretrained.models.InceptionV3Extractor.create(**kwargs)
    elif model == 'r3d_18':
        return features.pretrained.models.Resnet3d18Extractor.create(**kwargs)
    elif model == 'i3d':
        return features.pretrained.models.I3DExtractor.create(**kwargs)
    elif model == 'r3d_18_ap_mean':
        return features.pretrained.models.Resnet3d18MeanAPExtractor.create(**kwargs)
    elif model == 'i3d_mean':
        return features.pretrained.models.I3DMeanExtractor.create(**kwargs)
    else:
        raise RuntimeError(f'Unrecognized model {model}')

def composite_label_to_unique(composite_label):
    # If there is just a single label, return that.
    if '_' not in composite_label:
        return composite_label

    # For frequently-occurring combinations of labels, we create an entire class.
    pieces = set(composite_label.split('_'))
    label_combinations = [
        ['bedded', 'chewing'],
        ['bedded', 'foraging'],
        ['bedded', 'grooming'],
        ['bedded', 'looking around'],
    ]
    for label_combo in label_combinations:
        if pieces == set(label_combo):
            return '_'.join(label_combo)

    # If the combination of labels is not frequently occurring, pick the label that occurs less frequently.
    label_priorities = [
        'lie down',
        'stand up',
        'neutral',
        'snow',
        'grooming',
        'looking around',
        'travel',
        'chewing',
        'foraging',
        'bedded',
    ]
    for label in label_priorities:
        if label in composite_label:
            return label

def load_Xy(feature_name, featurestore, dbcon, vids, ignore_labels=[], adjust_feature_time=True, labels_fully_overlap=True,):
    nonaggregated_features = featurestore.get_nonaggregated_dataset(feature_name=feature_name, vids=vids)
    features_and_labels = featurestore.get_labels(nonaggregated_features, dbcon, adjust_feature_time=adjust_feature_time, full_overlap=labels_fully_overlap, ignore_labels=ignore_labels)

    if len(ignore_labels):
        # This isn't great because if a fid has an ignore_label well as some other label, we'll still filter it out.
        features_and_labels = features_and_labels.filter(
            pa.compute.invert(
                pa.compute.is_in(features_and_labels['labels'], pa.array(ignore_labels))
            )
        )

    X = np.vstack(features_and_labels.column('feature').to_numpy())
    y_composite = features_and_labels.column('labels').to_numpy()
    # Transform the composite labels that have all activities occurring in a frame to a single label.
    # y = np.array([composite_label_to_unique(y_comp) for y_comp in y_composite], dtype=object)
    # We don't have to coalesce to a single label when we're doing multi-label classification.
    y = y_composite
    return Dataset(
        X,
        y,
        y_composite,
        features_and_labels['start_time'].to_numpy(),
        features_and_labels['end_time'].to_numpy(),
        features_and_labels['fid'].to_numpy(),
        features_and_labels['vid'].to_numpy(),
    )
