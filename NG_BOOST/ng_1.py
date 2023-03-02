from ngboost import NGBClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import time
import os
import warnings
warnings.filterwarnings('ignore')

from ngboost import NGBClassifier

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

datasets = load_breast_cancer()

x = datasets.data
y = datasets['target']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


ngb = NGBClassifier().fit(x_train, y_train)
y_preds = ngb.predict(x_test)
y_dists = ngb.pred_dist(x_test)


# test Mean Squared Error
test_MSE = mean_squared_error(y_preds, y_test)
print('Test MSE', test_MSE)

