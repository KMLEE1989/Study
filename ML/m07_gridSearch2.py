import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

datasets = load_iris()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold
from sklearn.model_selection import cross_val_score, GridSearchCV

x_train, x_test,y_train,y_test=train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

# n_splits=5
# kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

# # parameter=[
# #     {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5]},     #12
# #     {"C":[1,10,100], "kernel":["rbf"], "gamma":[0.001,0.0001]},         #6
# #     {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.01,0.001,0.0001],"degree":[3,4]}   #24
# ]   #총 42개

#2. 모델구성
# model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True)
model = SVC(C=1, kernel='linear',degree=3)
# scores = cross_val_score(model, x, y, cv=kfold)
# print("ACC : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))
      
#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측

# x_test = x_train #과적합 상황 보여주기 
# y_test = y_train #  train 데이터로 best_estimator_로 예측뒤 점수를 내면
                   # best_score_ 나온다. 

#print("최적의 매개변수 : ", model.best_estimator_)
#print("최적의 파라미터 : ", model.best_params_)

#print("best_score_: ", model.best_score_)
print("model.score: ", model.score(x_test, y_test)) #score는 evaluate 개념

y_predict=model.predict(x_test)
print("accuracy_score: ", accuracy_score(y_test, y_predict))    

# best_score_:  0.9916666666666668 훈련값 그냥 accuracy
# model.score:  0.9666666666666667  테스트값  r accuracy
# accuracy_score:  0.9666666666666667 테스트값

# y_pred_best = model.best_estimator_.predict(x_test)
# print("최적 튠 AAC : ", accuracy_score(y_test,y_pred_best) )

##############################################################################################


# #print(model.cv_results_)
# aaa = pd.DataFrame(model.cv_results_)
# print(aaa)

# bbb=aaa[['params', 'mean_test_score','rank_test_score','split0_test_score']]
#    # 'split0_test_score','split1_test_score','split2_test_score',
#    # 'split3_test_score', 'split4_test_score'
#    # ]]
# print(bbb)