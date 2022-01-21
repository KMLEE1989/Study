import numpy as np
from sklearn.datasets import load_iris
from sklearn.svm import SVC
import pandas as pd
from sklearn.datasets import load_iris
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

datasets = load_iris()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.model_selection import cross_val_score, GridSearchCV

x_train, x_test,y_train,y_test=train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

parameter = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6,8,10,12], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3, 5, 7,10], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [2, 3, 5, 10]}]

#2. 모델구성
#model = GridSearchCV(SVC(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1)
model = RandomizedSearchCV(RandomForestClassifier(), parameter, cv=kfold, verbose=1, refit=True, n_jobs=-1, random_state=66) #20*5=100 #n_iter=20
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

"""
GridSearchCV
Fitting 5 folds for each of 44 candidates, totalling 220 fits
최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=10)
최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 10}
best_score_ :  0.9583333333333334
model.score:  0.9666666666666667
accuracy_score:  0.9666666666666667
최적 튠 ACC:  0.9666666666666667
걸린 시간:  7.452065706253052


========================================
RandomSearchCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
최적의 매개변수 :  RandomForestClassifier(min_samples_leaf=7, min_samples_split=10)
최적의 파라미터 :  {'min_samples_split': 10, 'min_samples_leaf': 7}
best_score_:  0.95
model.score:  0.9333333333333333
accuracy_score:  0.9333333333333333
최적 튠 AAC :  0.9333333333333333
걸린시간 :  2.076414108276367

"""