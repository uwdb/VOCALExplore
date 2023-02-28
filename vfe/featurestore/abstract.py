class AbstractFeatureStore:
    def __init__(self, *args, **kwargs):
        super().__init__()

    def insert_batch(self, feature_name, vids, starts, ends, features):
        pass
