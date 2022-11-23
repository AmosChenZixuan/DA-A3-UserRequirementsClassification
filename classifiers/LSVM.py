from .abstract import BaseClassifier
from sklearn.svm import LinearSVC

class LSVM(BaseClassifier):

    def name(self):
        return self.__class__.__name__

    @BaseClassifier.timer
    def model_selection(self, X, y):
        self.clf = LinearSVC(dual=False)
        regularization = [0.1, 1, 10, 100]

        grid = {
            'penalty': ['l1', 'l2'],
            'C': regularization, 
            'multi_class':['ovr', 'crammer_singer']
        }
        grid_search = self.grid_search(self.clf, grid)

        grid_search.fit(X,y)

        self.clf = grid_search.best_estimator_