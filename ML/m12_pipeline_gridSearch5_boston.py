import numpy as np 
from sklearn.datasets import load_boston
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, HalvingGridSearchCV
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#1) 데이터
datasets = load_boston()
#print(datasets.DESCR) # x = (150,4) y = (150,1)   ----> y를 (150,3)으로 바꿔줘야함
#print(datasets.feature_names)

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#### make_pipeline 사용시 ####

parameters = [
    {'randomforestregressor__max_depth' : [6, 8, 10], 'randomforestregressor__min_samples_leaf' : [3, 5, 7]},
    {'randomforestregressor__min_samples_leaf' : [3, 5, 7], 'randomforestregressor__min_samples_split' : [3, 5, 10]}]
# pipe 자리에는 사용할 model에 대한 parameters가 나와야하므로 (이 때 모델명은 소문자로 써줘야 한다. / _이거는 2번 써준다.)


#### Pipe 사용시 ####
# parameters = [
#     {'rf__max_depth' : [6, 8, 10], 'rf__min_samples_leaf' : [3, 5, 7]},
#     {'rf__min_samples_leaf' : [3, 5, 7], 'rf__min_samples_split' : [3, 5, 10]}]

#2) 모델구성 
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA  # 주성분분석 (고차원의 데이터를 저차원의 데이터로 축소시키는 차원 축소 방법)
                                       
pipe = make_pipeline(StandardScaler(), RandomForestRegressor())  
#pipe = Pipeline([("ss",StandardScaler()), ("rf", RandomForestRegressor())])  

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
make pipeline method
Grid
Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  11.277807474136353
model.score:  0.9218812456972526
accuracy_score:  0.9218812456972526

randomized CV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  6.404905080795288
model.score:  0.924143218549777
accuracy_score:  0.924143218549777

Halving
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 44
max_resources_: 404
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 18
n_resources: 44
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 1
n_candidates: 6
n_resources: 132
Fitting 5 folds for each of 6 candidates, totalling 30 fits
----------
iter: 2
n_candidates: 2
n_resources: 396
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  11.259912967681885
model.score:  0.9198869909042693
accuracy_score:  0.9198869909042693

Pipeline method

GRID
Fitting 5 folds for each of 18 candidates, totalling 90 fits
걸린 시간:  11.13618516921997
model.score:  0.92273529089419
accuracy_score:  0.92273529089419

randomizedCV
Fitting 5 folds for each of 10 candidates, totalling 50 fits
걸린 시간:  6.188418388366699
model.score:  0.9209233601982061
accuracy_score:  0.9209233601982061

Halving
n_iterations: 3
n_required_iterations: 3
n_possible_iterations: 3
min_resources_: 44
max_resources_: 404
aggressive_elimination: False
factor: 3
----------
iter: 0
n_candidates: 18
n_resources: 44
Fitting 5 folds for each of 18 candidates, totalling 90 fits
----------
iter: 1
n_candidates: 6
n_resources: 132
Fitting 5 folds for each of 6 candidates, totalling 30 fits
----------
iter: 2
n_candidates: 2
n_resources: 396
Fitting 5 folds for each of 2 candidates, totalling 10 fits
걸린 시간:  11.225948810577393
model.score:  0.9223547095882321
accuracy_score:  0.9223547095882321

"""