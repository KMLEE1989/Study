import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1) 데이터
datasets = load_diabetes()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#### make_pipeline 사용시 ####

parameters = [
    {'decisiontreeregressor__max_depth' : [6, 8, 10], 'decisiontreeregressor__min_samples_leaf' : [3, 5, 7]},
    {'decisiontreeregressor__min_samples_leaf' : [3, 5, 7], 'decisiontreeregressor__min_samples_split' : [3, 5, 10]}]
# pipe 자리에는 사용할 model에 대한 parameters가 나와야하므로 (이 때 모델명은 소문자로 써줘야 한다. / _이거는 2번 써준다.)


#### Pipe 사용시 ####
# parameters = [
#     {'dt__max_depth' : [6, 8, 10], 'dt__min_samples_leaf' : [3, 5, 7]},
#     {'dt__min_samples_leaf' : [3, 5, 7], 'dt__min_samples_split' : [3, 5, 10]}]

#2) 모델구성 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA  # 주성분분석 (고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법)
                                       
pipe = make_pipeline(StandardScaler(), DecisionTreeRegressor())  
#pipe = Pipeline([("ss",StandardScaler()), ("dt", DecisionTreeRegressor())])  

#model = GridSearchCV(pipe, parameters, cv=5, verbose=1) 
#model = RandomizedSearchCV(pipe, parameters, cv=5, verbose=1) 
model = HalvingGridSearchCV(pipe, parameters, cv=5, verbose=1)

#3) 훈련
import time

start = time.time()
model.fit(x_train, y_train) 
end = time.time()

#4) 평가, 예측
result = model.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict) 

print("걸린 시간: ", end - start)
print("model.score: ", result)
print("accuracy_score: ", r2)

"""
Make pippeline method

Grid
Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  0.18353724479675293
model.score:  0.15524116837935154
accuracy_score:  0.15524116837935154

Random
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  0.10375046730041504
model.score:  0.15524116837935154
accuracy_score:  0.15524116837935154

Halving
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 39
max_resources_: 353
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 18
n_resources: 39
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 1
n_candidates: 6
n_resources: 117
Fitting 5 folds for each of 6 candidates, totalling 30 fits
----------
iter: 2
n_candidates: 2
n_resources: 351
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  0.23141074180603027
model.score:  0.14972243882720404
accuracy_score:  0.14972243882720404

pipeline method

GRID
Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  0.18054556846618652
model.score:  0.15524116837935154
accuracy_score:  0.15524116837935154

RANDOM
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  0.10571670532226562
model.score:  0.15524116837935154
accuracy_score:  0.15524116837935154

Halving
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 39
max_resources_: 353
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 18
n_resources: 39
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 1
n_candidates: 6
n_resources: 117
Fitting 5 folds for each of 6 candidates, totalling 30 fits
----------
iter: 2
n_candidates: 2
n_resources: 351
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  0.23833250999450684
model.score:  0.14972243882720404
accuracy_score:  0.14972243882720404
"""