from .abstract import BaseClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

class DecisionTree(BaseClassifier):

    def name(self):
        return self.__class__.__name__

    @BaseClassifier.timer
    def model_selection(self, X, y):
        self.clf = DecisionTreeClassifier()
        depths = np.arange(10,21)
        num_leafs = [1, 5, 10, 20, 50, 100]
        num2split = [2, 4, 8, 16, 32]

        grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': depths,
            'min_samples_leaf': num_leafs,
            'min_samples_split': num2split,
        }
        grid_search = self.grid_search(self.clf, grid)

        grid_search.fit(X,y)

        self.clf = grid_search.best_estimator_