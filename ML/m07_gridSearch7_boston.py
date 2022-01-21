import numpy as np, pandas as pd
from sklearn.datasets import load_boston
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10], 'min_samples_leaf' : [3,5,7,10]},
    {'min_samples_leaf' : [3, 5, 7, 10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10], 'max_depth' : [6,8,10,12]}]

#2) 모델구성
# model = SVC(C=1, kernel = 'linear', degree=3)
model = GridSearchCV(RandomForestRegressor(), parameters, cv = kfold, verbose=1, 
                     refit=True, n_jobs=-1)  # cv = cross validation은 kfold로 / # 해당 모델에 맞는 parameter로 써주기!
                                                                # Fitting 5 folds for each of 42 candidates, totalling 210 fits
                                                                # refit = True : 가장 좋은 값을 뽑아내겠다.            
                                                                
#3) 훈련
import time
start = time.time()
model.fit(x_train, y_train)
end = time.time()

#4) 평가, 예측
print("최적의 매개변수: ", model.best_estimator_) 
print("최적의 파라미터: ", model.best_params_) 

print("best_score_ : ", model.best_score_)  
print("model.score: ", model.score(x_test, y_test)) 
 
y_predict = model.predict(x_test)
print("r2_score: ", r2_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 R2: ", r2_score(y_test, y_pred_best))  

print("걸린 시간: ", end - start)

# Fitting 5 folds for each of 56 candidates, totalling 280 fits
# 최적의 매개변수:  RandomForestRegressor(max_depth=10)
# 최적의 파라미터:  {'max_depth': 10, 'n_estimators': 100}
# best_score_ :  0.8362736360011601
# model.score:  0.9165279206899559
# r2_score:  0.9165279206899559
# 최적 튠 R2:  0.9165279206899559
# 걸린 시간:  7.736279487609863
                               

###################################################################

#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
#print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)
"""
                                               params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.818180               24           0.865428
1               {'max_depth': 6, 'n_estimators': 200}         0.822244               21           0.868118
2               {'max_depth': 8, 'n_estimators': 100}         0.826337               16           0.864508
3               {'max_depth': 8, 'n_estimators': 200}         0.828479               10           0.872316
4              {'max_depth': 10, 'n_estimators': 100}         0.836274                1           0.878318
5              {'max_depth': 10, 'n_estimators': 200}         0.831103                6           0.871902
6              {'max_depth': 12, 'n_estimators': 100}         0.828969                9           0.874364
7              {'max_depth': 12, 'n_estimators': 200}         0.832347                3           0.869772
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.796944               32           0.806284
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.786893               40           0.770062
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.781591               44           0.762877
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.766877               56           0.734828
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.808779               26           0.813667
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.789281               35           0.780082
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.785334               42           0.768734
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.775008               50           0.751039
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.804857               27           0.803354
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.793565               33           0.773094
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.778509               46           0.749458
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.772761               53           0.741530
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.801938               30           0.804319
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.788519               38           0.772144
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.777043               48           0.750707
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.777626               47           0.759760
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.802067               29           0.792759
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.803098               28           0.797586
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.809752               25           0.816130
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.798773               31           0.794802
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.788618               37           0.767398
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.791881               34           0.773785
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.787400               39           0.769231
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.788819               36           0.769079
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.782259               43           0.757779
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.774683               51           0.756193
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.778665               45           0.758330
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.785583               41           0.775308
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.772552               54           0.749196
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.776822               49           0.750214
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.774325               52           0.757499
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.771890               55           0.744652
40           {'max_depth': 6, 'min_samples_split': 2}         0.821287               22           0.867144
41           {'max_depth': 6, 'min_samples_split': 3}         0.819896               23           0.857761
42           {'max_depth': 6, 'min_samples_split': 5}         0.827340               13           0.869697
43          {'max_depth': 6, 'min_samples_split': 10}         0.825519               18           0.867740
44           {'max_depth': 8, 'min_samples_split': 2}         0.823128               20           0.872541
45           {'max_depth': 8, 'min_samples_split': 3}         0.823721               19           0.859495
46           {'max_depth': 8, 'min_samples_split': 5}         0.828378               11           0.869418
47          {'max_depth': 8, 'min_samples_split': 10}         0.830022                7           0.866644
48          {'max_depth': 10, 'min_samples_split': 2}         0.825845               17           0.868385
49          {'max_depth': 10, 'min_samples_split': 3}         0.831239                5           0.878689
50          {'max_depth': 10, 'min_samples_split': 5}         0.829233                8           0.876424
51         {'max_depth': 10, 'min_samples_split': 10}         0.826539               15           0.873281
52          {'max_depth': 12, 'min_samples_split': 2}         0.827848               12           0.871652
53          {'max_depth': 12, 'min_samples_split': 3}         0.836245                2           0.883485
54          {'max_depth': 12, 'min_samples_split': 5}         0.827044               14           0.868140
55         {'max_depth': 12, 'min_samples_split': 10}         0.831822                4           0.871611
"""