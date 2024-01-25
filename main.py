from GBModel import DecisionTreeRegressor
from GBModel import Metrics

y_true = [1, 1, 1, 1, 1]

model = DecisionTreeRegressor()

y_pred = model.predict(y_true)


print(Metrics.mean_squared_error(y_true, y_pred))