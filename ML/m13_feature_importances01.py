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

model1=DecisionTreeClassifier(max_depth=5)
#model1=RandomForestClassifier(max_depth=5)
#model1=XGBClassifier(max_depth=5)
#model1=GradientBoostingClassifier(max_depth=5)


model1.fit(x_train, y_train)

result = model1.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model1.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("accuracy_score : ", acc)

print(model1.feature_importances_)

# GradientBoostingClassifier
# accuracy_score :  0.9666666666666667
# [0.00545879 0.01141205 0.17602845 0.80710071]

# XGBClassifier
# accuracy_score :  0.9
# [0.01835513 0.0256969  0.6204526  0.33549538]

# RandomForestClassifier
# accuracy_score :  0.9333333333333333
# [0.10795058 0.03810176 0.38507653 0.46887114]


