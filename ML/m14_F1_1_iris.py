#실습
#피처임포턴스가 전체 중요도에서 하위 20~25% 컬럼들을 제거하여
#데이터셋 재구성후
#각 모델별로 돌려서 결과 도출!

# 기존 모델결과와 비교 

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import pandas as pd

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

#x=pd.DataFrame(x)
# x=x.drop([0],axis=1)
# x=x.drop([1],axis=1)
x = np.delete(x,[0,1], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

#model1=DecisionTreeClassifier(max_depth=5)
#model1=RandomForestClassifier(max_depth=5)
#model1=XGBClassifier(max_depth=5)
model1=GradientBoostingClassifier(max_depth=5)

model1.fit(x_train, y_train)

result = model1.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model1.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("accuracy_score : ", acc)

print(model1.feature_importances_)

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
    
plot_feature_importances_datasets(model1)
plt.show()


#결과비교
#1.DecisionTree
#기존 acc : 0.9666666666666667
#컬럼삭제후 acc : 0.9333333333333333

#2.RandomForest
#기존 acc : 0.9333333333333333
#컬럼삭제후 acc :  0.9666666666666667

#3.XGB
#기존 acc : 0.9
#컬럼삭제후 acc: 0.9666666666666667

#4.GradientBoosting
#기존 acc : 0.9333333333333333
#컬럼삭제후 acc:  0.9333333333333333
