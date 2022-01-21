import numpy as np, pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_diabetes()
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
# 최적의 매개변수:  RandomForestRegressor(min_samples_leaf=3, min_samples_split=5)
# 최적의 파라미터:  {'min_samples_leaf': 3, 'min_samples_split': 5}
# best_score_ :  0.503022143545943
# model.score:  0.37227629328770717
# r2_score:  0.37227629328770717
# 최적 튠 R2:  0.37227629328770717
# 걸린 시간:  6.791804552078247

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
0               {'max_depth': 6, 'n_estimators': 100}         0.490761               26           0.490441
1               {'max_depth': 6, 'n_estimators': 200}         0.492163               19           0.498578
2               {'max_depth': 8, 'n_estimators': 100}         0.493139               14           0.492676
3               {'max_depth': 8, 'n_estimators': 200}         0.495053                8           0.500290
4              {'max_depth': 10, 'n_estimators': 100}         0.489046               34           0.484837
5              {'max_depth': 10, 'n_estimators': 200}         0.484980               41           0.473369
6              {'max_depth': 12, 'n_estimators': 100}         0.481793               51           0.470272
7              {'max_depth': 12, 'n_estimators': 200}         0.484824               43           0.483834
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.493868               11           0.491669
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.493476               13           0.525417
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.489685               30           0.517994
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.485438               39           0.516105
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.490063               29           0.515214
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.497687                2           0.513518
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.491615               20           0.517370
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.486795               37           0.516466
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.493022               16           0.505758
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.489272               33           0.506732
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.490584               27           0.528540
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.482664               48           0.525699
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.495560                7           0.510064
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.494297                9           0.511186
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.491300               22           0.518329
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.489374               32           0.523737
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.496392                4           0.522149
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.490948               24           0.525932
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.503022                1           0.504333
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.494061               10           0.518939
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.491488               21           0.510911
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.489603               31           0.493938
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.496181                5           0.527088
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.492719               17           0.529432
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.482153               49           0.519677
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.480211               55           0.506322
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.484890               42           0.520183
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.490564               28           0.508019
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.483256               47           0.519130
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.481532               52           0.509811
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.485221               40           0.518248
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.486453               38           0.516820
40           {'max_depth': 6, 'min_samples_split': 2}         0.488915               35           0.491704
41           {'max_depth': 6, 'min_samples_split': 3}         0.483582               46           0.482478
42           {'max_depth': 6, 'min_samples_split': 5}         0.490970               23           0.502545
43          {'max_depth': 6, 'min_samples_split': 10}         0.483971               44           0.497152
44           {'max_depth': 8, 'min_samples_split': 2}         0.478743               56           0.447171
45           {'max_depth': 8, 'min_samples_split': 3}         0.480636               54           0.487436
46           {'max_depth': 8, 'min_samples_split': 5}         0.482102               50           0.476025
47          {'max_depth': 8, 'min_samples_split': 10}         0.497298                3           0.506083
48          {'max_depth': 10, 'min_samples_split': 2}         0.493782               12           0.523231
49          {'max_depth': 10, 'min_samples_split': 3}         0.483600               45           0.473177
50          {'max_depth': 10, 'min_samples_split': 5}         0.493113               15           0.493283
51         {'max_depth': 10, 'min_samples_split': 10}         0.492312               18           0.510103
52          {'max_depth': 12, 'min_samples_split': 2}         0.495795                6           0.503061
53          {'max_depth': 12, 'min_samples_split': 3}         0.481120               53           0.470896
54          {'max_depth': 12, 'min_samples_split': 5}         0.490786               25           0.503101
55         {'max_depth': 12, 'min_samples_split': 10}         0.486820               36           0.479693
"""