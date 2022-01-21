import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
import pandas as pd
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

datasets = load_breast_cancer()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

x_train, x_test,y_train,y_test=train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter=[
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'max_depth' : [6,8,10,12]}
]   #총 42개

#2. 모델구성
#model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1)
#model = RandomizedSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66, n_iter=20) #20*5=100

model = HalvingGridSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1)

# model = SVC(C=1, kernel='linear',degree=3)
# scores = cross_val_score(model, x, y, cv=kfold)
# print("ACC : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))
      
#3. 훈련
import time
start= time.time()
model.fit(x_train, y_train)
end =time.time()

#4. 평가, 예측

# x_test = x_train #과적합 상황 보여주기 
# y_test = y_train #  train 데이터로 best_estimator_로 예측뒤 점수를 내면
                   # best_score_ 나온다. 

print("최적의 매개변수 : ", model.best_estimator_)
print("최적의 파라미터 : ", model.best_params_)

print("best_score_: ", model.best_score_)
print("model.score: ", model.score(x_test, y_test)) #score는 evaluate 개념

y_predict=model.predict(x_test)
print("accuracy_score: ", accuracy_score(y_test, y_predict))    

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 AAC : ", accuracy_score(y_test,y_pred_best))

print("걸린시간 : ", end-start)




'''
Grid
Fitting 5 folds for each of 44 candidates, totalling 220 fits
최적의 매개변수:  RandomForestClassifier(max_depth=10)
최적의 파라미터:  {'max_depth': 10, 'n_estimators': 100}
best_score_ :  0.9692307692307691
model.score:  0.9649122807017544
accuracy_score:  0.9649122807017544
최적 튠 ACC:  0.9649122807017544
걸린 시간:  9.314090251922607

Random
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 :  RandomForestClassifier(max_depth=10, n_estimators=200)
최적의 파라미터 :  {'n_estimators': 200, 'max_depth': 10}
best_score_:  0.9604395604395606
model.score:  0.9649122807017544
accuracy_score:  0.9649122807017544
최적 튠 AAC :  0.9649122807017544
걸린시간 :  2.8563623428344727

Halving
n_iterations: 3
n_required_iterations: 4
n_possible_iterations: 3
min_resources_: 20
max_resources_: 455
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 56
n_resources: 20
Fitting 5 folds for each of 56 candidates, totalling 280 fits
----------
iter: 1
n_candidates: 19
n_resources: 60
Fitting 5 folds for each of 19 candidates, totalling 95 fits
----------
iter: 2
n_candidates: 7
n_resources: 180
Fitting 5 folds for each of 7 candidates, totalling 35 fits
최적의 매개변수 :  RandomForestClassifier(max_depth=8)
최적의 파라미터 :  {'max_depth': 8, 'n_estimators': 100}
best_score_:  0.9666666666666668
model.score:  0.9736842105263158
accuracy_score:  0.9736842105263158
최적 튠 AAC :  0.9736842105263158
걸린시간 :  12.198824644088745
'''

##############################################################################################
