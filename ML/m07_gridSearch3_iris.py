#실습


#모델: RandomForestClassifier

# parameters=[{'n_estimators' : [100,200]},
#             {'max_depth : [6,8,10,12]}'},
#             {'min_samples_leaf': [3,5,7,10]},
#             {'min_samples_split' : [2,3,5,10]},
#             {'n_jobs':[-1,2,4]}    
# ]

#파라미터 조합으로 2개이상 엮을것

import numpy as np, pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score
#1) 데이터
datasets = load_iris()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [100, 200], 'min_samples_leaf' : [6, 8, 10, 12]},
    {'min_samples_leaf' : [3, 5, 7,10], 'min_samples_split' : [2, 3, 5, 10]},
    {'n_jobs' : [2, 3, 5, 10]}]

#2) 모델구성
# model = SVC(C=1, kernel = 'linear', degree=3)
model = GridSearchCV(RandomForestClassifier(), parameters, cv = kfold, verbose=1, 
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
print("accuracy_score: ", accuracy_score(y_test, y_predict)) 

y_pred_best = model.best_estimator_.predict(x_test)
print("최적 튠 ACC: ", accuracy_score(y_test, y_pred_best))  

print("걸린 시간: ", end - start)

"""
Fitting 5 folds for each of 44 candidates, totalling 220 fits
최적의 매개변수:  RandomForestClassifier(max_depth=6, min_samples_leaf=10)
최적의 파라미터:  {'max_depth': 6, 'min_samples_leaf': 10}
best_score_ :  0.9583333333333334
model.score:  0.9666666666666667
accuracy_score:  0.9666666666666667
최적 튠 ACC:  0.9666666666666667
걸린 시간:  7.452065706253052
"""

###################################################################

#print(model.cv_results_)    # 'mean_fit_time': 평균 훈련 시간(42번)
aaa = pd.DataFrame(model.cv_results_)  # 보기 편하게 하기 위해 DataFrame시켜줌
print(aaa)

bbb = aaa[['params','mean_test_score','rank_test_score','split0_test_score']]
     #'split1_test_score', 'split2_test_score',
     #'split3_test_score','split4_test_score']]  # split0_test_score = kfold가 5개이므로..

print(bbb)

'''
   mean_fit_time  std_fit_time  mean_score_time  std_score_time param_max_depth  ... split3_test_score split4_test_score mean_test_score std_test_score  rank_test_score
0        0.163563      0.019401         0.007779    7.463913e-04               6  ...               1.0             0.875        0.950000       0.040825                2
1        0.358043      0.016866         0.022739    1.507822e-02               6  ...               1.0             0.875        0.950000       0.040825                2
2        0.172738      0.015875         0.028723    1.748503e-02               8  ...               1.0             0.875        0.941667       0.042492               19
3        0.329718      0.002239         0.013165    3.987074e-04               8  ...               1.0             0.875        0.950000       0.040825                2
4        0.155813      0.014149         0.013335    1.278065e-02              10  ...               1.0             0.875        0.941667       0.042492               19
5        0.323535      0.030359         0.022739    1.184081e-02              10  ...               1.0             0.875        0.950000       0.040825                2
6        0.149999      0.036653         0.006981    9.536743e-08              12  ...               1.0             0.875        0.950000       0.040825                2
7        0.267086      0.050317         0.024933    1.390577e-02              12  ...               1.0             0.875        0.950000       0.040825                2
8        0.135238      0.027880         0.014562    1.079092e-02               6  ...               1.0             0.875        0.941667       0.042492               19
9        0.132047      0.026316         0.016157    1.061823e-02               6  ...               1.0             0.875        0.950000       0.040825                2
10       0.150997      0.034952         0.009374    3.313700e-03               6  ...               1.0             0.875        0.941667       0.042492               19
11       0.135637      0.030565         0.013364    1.278141e-02               6  ...               1.0             0.875        0.958333       0.045644                1
12       0.134640      0.026167         0.006981    1.168008e-07               8  ...               1.0             0.875        0.950000       0.040825                2
13       0.136237      0.032572         0.007380    4.886168e-04               8  ...               1.0             0.875        0.950000       0.048591                2
14       0.155784      0.030743         0.006981    2.780415e-07               8  ...               1.0             0.875        0.941667       0.042492               19
15       0.136646      0.025184         0.019137    1.410590e-02               8  ...               1.0             0.875        0.941667       0.042492               19
16       0.129055      0.023318         0.019747    1.197476e-02              10  ...               1.0             0.875        0.941667       0.042492               19
17       0.134640      0.024905         0.007380    4.886945e-04              10  ...               1.0             0.875        0.950000       0.048591                2
18       0.144613      0.011951         0.018151    1.251084e-02              10  ...               1.0             0.875        0.941667       0.042492               19
19       0.118882      0.031948         0.012567    8.141667e-03              10  ...               1.0             0.875        0.950000       0.048591                2
20       0.134041      0.027829         0.006782    3.985883e-04              12  ...               1.0             0.875        0.941667       0.042492               19
21       0.140425      0.027345         0.007579    4.883052e-04              12  ...               1.0             0.875        0.941667       0.042492               19
22       0.144018      0.034223         0.014159    1.238438e-02              12  ...               1.0             0.875        0.941667       0.042492               19
23       0.140424      0.030620         0.006782    3.988743e-04              12  ...               1.0             0.875        0.950000       0.040825                2
24       0.131648      0.027284         0.019747    1.146557e-02             NaN  ...               1.0             0.875        0.950000       0.040825                2
25       0.137831      0.031979         0.012168    9.881273e-03             NaN  ...               1.0             0.875        0.941667       0.042492               19
26       0.131049      0.030083         0.011968    9.482638e-03             NaN  ...               1.0             0.875        0.941667       0.042492               19
27       0.127459      0.027706         0.019348    1.514622e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
28       0.132047      0.026134         0.018151    1.382493e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
29       0.134839      0.029729         0.007181    3.988028e-04             NaN  ...               1.0             0.875        0.941667       0.042492               19
30       0.127459      0.025895         0.013165    1.187448e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
31       0.143217      0.031175         0.007180    3.983975e-04             NaN  ...               1.0             0.875        0.941667       0.042492               19
32       0.131847      0.029555         0.011370    9.283294e-03             NaN  ...               1.0             0.875        0.941667       0.042492               19
33       0.132645      0.030895         0.013963    1.247258e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
34       0.126858      0.023702         0.012167    1.039140e-02             NaN  ...               1.0             0.875        0.950000       0.048591                2
35       0.136236      0.029218         0.012566    1.118763e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
36       0.138230      0.028827         0.007181    3.987790e-04             NaN  ...               1.0             0.875        0.941667       0.042492               19
37       0.135637      0.031317         0.013763    1.307025e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
38       0.161768      0.028722         0.006981    6.306006e-04             NaN  ...               1.0             0.875        0.941667       0.042492               19
39       0.136834      0.029052         0.017952    1.176692e-02             NaN  ...               1.0             0.875        0.941667       0.042492               19
40       0.141621      0.031883         0.006981    6.311281e-04             NaN  ...               1.0             0.875        0.950000       0.040825                2
41       0.131847      0.025336         0.010771    7.091511e-03             NaN  ...               1.0             0.875        0.950000       0.040825                2
42       0.130850      0.030345         0.014162    1.386856e-02             NaN  ...               1.0             0.875        0.950000       0.040825                2
43       0.117885      0.023630         0.009575    2.410570e-03             NaN  ...               1.0             0.875        0.941667       0.042492               19

[44 rows x 17 columns]
                                               params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.950000                2           0.958333
1               {'max_depth': 6, 'n_estimators': 200}         0.950000                2           0.958333
2               {'max_depth': 8, 'n_estimators': 100}         0.941667               19           0.958333
3               {'max_depth': 8, 'n_estimators': 200}         0.950000                2           0.958333
4              {'max_depth': 10, 'n_estimators': 100}         0.941667               19           0.958333
5              {'max_depth': 10, 'n_estimators': 200}         0.950000                2           0.958333
6              {'max_depth': 12, 'n_estimators': 100}         0.950000                2           0.958333
7              {'max_depth': 12, 'n_estimators': 200}         0.950000                2           0.958333
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.941667               19           0.958333
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.950000                2           0.958333
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.941667               19           0.958333
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.958333                1           0.958333
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.950000                2           0.958333
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.950000                2           0.958333
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.941667               19           0.958333
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.941667               19           0.958333
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.941667               19           0.958333
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.950000                2           0.958333
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.941667               19           0.958333
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.950000                2           0.958333
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.941667               19           0.958333
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.941667               19           0.958333
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.941667               19           0.958333
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.950000                2           0.958333
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.950000                2           0.958333
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.941667               19           0.958333
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.941667               19           0.958333
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.941667               19           0.958333
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.941667               19           0.958333
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.941667               19           0.958333
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.941667               19           0.958333
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.941667               19           0.958333
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.941667               19           0.958333
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.950000                2           0.958333
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.941667               19           0.958333
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.941667               19           0.958333
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.941667               19           0.958333
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.941667               19           0.958333
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.941667               19           0.958333
40                           {'min_samples_split': 2}         0.950000                2           0.958333
41                           {'min_samples_split': 3}         0.950000                2           0.958333
42                           {'min_samples_split': 5}         0.950000                2           0.958333
43                          {'min_samples_split': 10}         0.941667               19           0.958333
'''