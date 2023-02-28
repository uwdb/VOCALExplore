class AbstractLabeler:
    def __init__(self, y_true):
        self.y_true = y_true

    def label(self, X, X_indices):
        raise NotImplementedError

    @classmethod
    def name(cls):
        raise NotImplementedError
