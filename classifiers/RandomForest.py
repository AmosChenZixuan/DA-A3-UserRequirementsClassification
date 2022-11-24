from .abstract import BaseClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np

class RandomForest(BaseClassifier):

    def name(self):
        return self.__class__.__name__

    @BaseClassifier.timer
    def default_model(self, X, y):
        clf = RandomForestClassifier()
        clf.fit(X, y)
        self.clf = clf

    @BaseClassifier.timer
    def model_selection(self, X, y):
        self.clf = RandomForestClassifier()
        n_clf = np.linspace(start = 100, stop = 1000, num = 10, dtype=int)
        depths = list(np.arange(10,51)) + [None]
        num_leafs = np.arange(1,21)
        num2split = np.arange(2,21)

        grid = {
            'n_estimators': n_clf,
            'criterion': ['gini', 'entropy'],
            'max_depth': depths,
            'min_samples_leaf': num_leafs,
            'min_samples_split': num2split,
        }
        random_search = self.random_search(self.clf, grid)

        random_search.fit(X,y)

        self.clf = random_search.best_estimator_