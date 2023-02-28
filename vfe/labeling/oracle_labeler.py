from .abstractlabeler import AbstractLabeler

class OracleLabeler(AbstractLabeler):
    def label(self, X, X_indices):
        return self.y_true[X_indices]

    @classmethod
    def name(cls):
        return 'oracle'
