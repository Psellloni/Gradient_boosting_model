# Libraries
import numpy as np

# Metrics
class Metrics:

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

class DecisionTreeRegressor:

    def fit(self, X, y):
        pass

    def predict(self, X):

        return np.ones(len(X))