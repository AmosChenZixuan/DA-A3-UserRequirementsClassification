from .abstract import BaseClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

class AdaBoost(BaseClassifier):

    def name(self):
        return self.__class__.__name__

    @BaseClassifier.timer
    def model_selection(self, X, y):
        self.clf = AdaBoostClassifier(algorithm = 'SAMME')
        base_estimator = [LinearSVC(C=100, dual=False), 
            DecisionTreeClassifier(max_depth=10)]

        learning_rate = [1., 0.1, 0.01]

        grid = {
            'base_estimator': base_estimator,
            'learning_rate': learning_rate, 
        }
        grid_search = self.grid_search(self.clf, grid)

        grid_search.fit(X,y)

        self.clf = grid_search.best_estimator_