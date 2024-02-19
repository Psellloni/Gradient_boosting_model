from DecisionTreeClassifier import DecisionTreeClassifier as DTC
import pandas as pd

x_train = [[-1, 1, 2, 2, 1, -1, -1, -1, -1], [-4, -3, -1, 1, 2, 3, 5, 7, 9]]
y_train = [[0, 0, 1, 1, 1, 1, 0, 0, 0]]

x_test = [[0, 4, 0], [-8, 0, 12]]
y_test = [0, 1, 0]



model = DTC()
model.fit(x_train, y_train)
model.print_tree