import numpy as np
from sklearn.datasets import load_diabetes
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

datasets = load_diabetes()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV

x_train, x_test,y_train,y_test=train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter=[{'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10] },
    {'min_samples_leaf' : [3, 5, 7,10], 'min_samples_split' : [2, 3, 5, 10]}
    # {"C":[1,10,100,1000], "kernel":["linear"], "degree":[3,4,5,6,7]},     #20
    # {"C":[1,10,100,1000], "kernel":["rbf"], "gamma":[0.001,0.0001,0.00001]},         #12
    # {"C":[1,10,100,1000], "kernel":["sigmoid"], "gamma":[0.01,0.001,0.0001],"degree":[3,4,5]}   #36
] 

#2. 모델구성
#model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1)
#model = RandomizedSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66, n_iter=20) #20*5=100

model = HalvingGridSearchCV(RandomForestRegressor(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1)

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
print("r2 score: ", r2_score(y_test, y_predict))    

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 AAC : ", r2_score(y_test,y_pred_best))

print("걸린시간 : ", end-start)



'''
GridSearchCV
Fitting 5 folds for each of 56 candidates, totalling 280 fits
최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=10)
최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 10}
best_score_ :  0.005714285714285714
model.score:  0.02247191011235955
accuracy_score:  0.02247191011235955
최적 튠 accuracy:  0.02247191011235955
걸린 시간:  8.883268594741821

=================================
RandomizedSearchCV (GridSearchCV에 비해 속도가 빠르다 / 성능은 유사) : Fitting 5 folds for each of 10 candidates, totalling 50 fits : 10번만 돈다(100번 이상중에 10개만 뽑아서 돈다.)
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수:  RandomForestRegressor(max_depth=12, min_samples_leaf=7)
최적의 파라미터:  {'min_samples_leaf': 7, 'max_depth': 12}
best_score_ :  0.4965434238961402
model.score:  0.40559563081576255
accuracy_score:  0.40559563081576255
최적 튠 ACC:  0.40559563081576255
걸린 시간:  2.4274792671203613


halving
n_iterations: 4
n_required_iterations: 4
n_possible_iterations: 4
min_resources_: 13
max_resources_: 353
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 40
n_resources: 13
Fitting 5 folds for each of 40 candidates, totalling 200 fits
----------
iter: 1
n_candidates: 14
n_resources: 39
Fitting 5 folds for each of 14 candidates, totalling 70 fits
----------
iter: 2
n_candidates: 5
n_resources: 117
Fitting 5 folds for each of 5 candidates, totalling 25 fits
----------
iter: 3
n_candidates: 2
n_resources: 351
Fitting 5 folds for each of 2 candidates, totalling 10 fits
최적의 매개변수 :  RandomForestRegressor(max_depth=10, min_samples_leaf=5)
최적의 파라미터 :  {'max_depth': 10, 'min_samples_leaf': 5}
best_score_:  0.49513557066153513
model.score:  0.38555175615067605
r2 score:  0.38555175615067605
최적 튠 AAC :  0.38555175615067605
걸린시간 :  7.67792010307312
'''
