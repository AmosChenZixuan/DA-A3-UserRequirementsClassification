from .abstract import BaseClassifier
import xgboost as xgb
from scipy.stats import uniform, randint

class XGBoost(BaseClassifier):

    def name(self):
        return self.__class__.__name__

    @BaseClassifier.timer
    def model_selection(self, X, y):
        self.clf = xgb.XGBClassifier()
        

        grid = {
            "colsample_bytree": uniform(0.7, 0.3),
            "gamma": uniform(0, 0.5),
            "learning_rate": uniform(0.03, 0.3),     # default 0.1 
            "max_depth": randint(2, 6),              # default 3
            "n_estimators": randint(100, 150),       # default 100
            "subsample": uniform(0.6, 0.4)
        }
        random_search = self.random_search(self.clf, grid)

        random_search.fit(X,y)

        self.clf = random_search.best_estimator_