import itertools
import numpy as np

def _to_list(x):
    return list(itertools.chain.from_iterable([
        c if hasattr(c, '__iter__') else [c]
        for c in x.values.tolist()
    ]))

def flatten_features(df):
    bad_columns = ['label', 'segment_id', 'object_id', 'idx']
    # Specify dtype=object to avoid a warning when the lists are of different lengths.
    return np.array(
            df[[c for c in df.columns if c not in bad_columns]].apply(
                _to_list,
                axis=1
        ).to_list(), dtype=object)
