from .abstract import AbstractStrategy

class FullySupervisedStrategy(AbstractStrategy):
    @classmethod
    def name(cls):
        return 'fullysupervised'

    def process_it(self, X_train, y_train, X_test, y_test, desired_count, stratify_train=None, stratify_test=None):
        num_done = self.num_done()
        print(f'Num done: {num_done}')
        if num_done >= desired_count:
            return
        results = self.train_and_evaluate_pytorch_model(X_train, y_train, X_test, y_test, it=num_done, stratify_test=stratify_test)
        self.save_results([results])
