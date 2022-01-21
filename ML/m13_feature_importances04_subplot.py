# 맹그러봐 subplot 이용해서 4개의 모델을 한 화면에 그래프로!

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
model2=RandomForestClassifier(max_depth=5)
model3=XGBClassifier(max_depth=5)
model4=GradientBoostingClassifier(max_depth=5)

model1.fit(x_train, y_train)
model2.fit(x_train, y_train)
model3.fit(x_train, y_train)
model4.fit(x_train, y_train)

#result = model1.score(x_test, y_test) 

# from sklearn.metrics import accuracy_score
# y_predict=model1.predict(x_test)
# acc = accuracy_score(y_test, y_predict)

#print("accuracy_score : ", acc)

#$print(model1.feature_importances_)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_datasets(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align='center')
    plt.yticks(np.arange(n_features),datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plt.subplot(2,2,1)
plot_feature_importances_datasets(model1)
plt.subplot(2,2,2)
plot_feature_importances_datasets(model2)
plt.subplot(2,2,3)
plot_feature_importances_datasets(model3)
plt.subplot(2,2,4)
plot_feature_importances_datasets(model4)

# plt.subplot(1,4,1)
# plot_feature_importances_datasets(model1)
# plt.subplot(1,4,2)
# plot_feature_importances_datasets(model2)
# plt.subplot(1,4,3)
# plot_feature_importances_datasets(model3)
# plt.subplot(1,4,4)
# plot_feature_importances_datasets(model4)

    
# plot_feature_importances_datasets(model1)
plt.show()

# accuracy_score :  0.9333333333333333
# [0.00944523 0.00689926 0.21839388 0.76526162]

"""
model_list = [model_1,model_2,model_3,model_4]
model_name = ['DecisionTreeClassifier','RandomForestClassifier','XGBClassifier','GradientBoostingClassifier']
for i in range(4):
    plt.subplot(2, 2, i+1)               # nrows=2, ncols=1, index=1
    model_list[i].fit(x_train, y_train)

    result = model_list[i].score(x_test, y_test)
    feature_importances_ = model_list[i].feature_importances_

    y_predict = model_list[i].predict(x_test)
    acc = accuracy_score(y_test, y_predict)
    print("result", result)
    print("accuracy_score", acc)
    print("feature_importances_", feature_importances_)
    plot_feature_importances_dataset(model_list[i])
    plt.ylabel(model_name[i])

plt.show()
"""