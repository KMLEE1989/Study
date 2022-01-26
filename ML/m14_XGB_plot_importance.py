from pyexpat import model
import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from xgboost import XGBClassifier, XGBRFClassifier

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#model1=DecisionTreeClassifier(max_depth=5)
#model1=RandomForestClassifier(max_depth=5)
model1=XGBClassifier(max_depth=5)
#model1=GradientBoostingClassifier(max_depth=5)

model1.fit(x_train, y_train)

result = model1.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model1.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("accuracy_score : ", acc)

print(model1.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

# def plot_feature_importances_datasets(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#              align='center')
#     plt.yticks(np.arange(n_features),datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)
    
# plot_feature_importances_datasets(model1)
# plt.show()

from xgboost.plotting import plot_importance

plot_importance(model1)
plt.show()


