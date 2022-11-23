from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from time import perf_counter

class BaseClassifier(ABC):
    def __init__(self):
        self.clf = None

    @abstractmethod
    def model_selection(self, X, y):
        pass

    @abstractmethod
    def name(self):
        pass

    def existsClf(func):
        def wrap(self, *args, **kwargs):
            if self.clf:
                return func(self, *args, **kwargs)
            else:
                raise AttributeError(f"You should first train a classifier with {self.name()}.model_selection. ")
        return wrap 

    def timer(func):
        def wrap(self, *args, **kwargs):
            start = perf_counter()
            ret = func(self, *args, **kwargs)
            end = perf_counter()
            print(f"Elapsed Time: {round(end - start, 4)} seconds")
            return ret
        return wrap

    @existsClf
    @timer
    def fit(self, X, y):
        self.clf.fit(X, y)

            
    @existsClf
    def predict(self, X):
        return self.clf.predict(X)

    def grid_search(self, clf, grid, cv=3, scoring='accuracy'):
        return GridSearchCV(clf, grid, \
                cv=StratifiedKFold(cv), \
                scoring=scoring, return_train_score=True, \
                n_jobs=-1)

    def random_search(self, clf, grid, cv=3, scoring='accuracy', n_iter=100):
        return RandomizedSearchCV(clf, grid, \
                cv=StratifiedKFold(cv), \
                scoring=scoring, return_train_score=True, \
                n_iter=n_iter, n_jobs=-1)

    @existsClf
    def get_params(self):
        return self.clf.get_params()



    