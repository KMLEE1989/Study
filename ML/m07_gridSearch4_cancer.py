import numpy as np, pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV   # 교차검증

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

n_splits = 5
kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)

parameters = [
    {'n_estimators' : [100, 200], 'max_depth' : [6, 8, 10, 12], 'max_depth' : [6, 8, 10, 12]},
    {'max_depth' : [6, 8, 10, 12],'min_samples_leaf' : [3, 5, 7,10]},
    {'min_samples_leaf' : [3, 5, 7,10], 'min_samples_split' : [2, 3, 5, 10]},
    {'min_samples_split' : [2, 3, 5, 10]}]

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
최적의 매개변수:  RandomForestClassifier(max_depth=10)
최적의 파라미터:  {'max_depth': 10, 'n_estimators': 100}
best_score_ :  0.9692307692307691
model.score:  0.9649122807017544
accuracy_score:  0.9649122807017544
최적 튠 ACC:  0.9649122807017544
걸린 시간:  9.314090251922607
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
0        0.235370      0.031865         0.013165    3.301716e-03               6  ...          0.956044          0.967033        0.960440       0.016447                5
1        0.385569      0.092535         0.029521    1.550996e-02               6  ...          0.945055          0.967033        0.958242       0.025441               11
2        0.208642      0.078369         0.010572    3.710358e-03               8  ...          0.945055          0.967033        0.956044       0.025059               16
3        0.412497      0.087687         0.019946    8.509604e-03               8  ...          0.945055          0.967033        0.960440       0.022628                5
4        0.186900      0.054292         0.007580    7.979036e-04              10  ...          0.967033          0.967033        0.969231       0.017582                1
5        0.370809      0.062552         0.032513    1.358150e-02              10  ...          0.956044          0.967033        0.960440       0.022628                5
6        0.180916      0.040887         0.016955    1.183454e-02              12  ...          0.978022          0.967033        0.967033       0.021978                3
7        0.367816      0.071493         0.019348    6.326492e-03              12  ...          0.956044          0.967033        0.958242       0.018906               11
8        0.175131      0.034958         0.012966    1.098004e-02               6  ...          0.945055          0.967033        0.956044       0.026005               16
9        0.193682      0.039262         0.007580    7.976890e-04               6  ...          0.945055          0.956044        0.953846       0.021308               27
10       0.202259      0.030009         0.020944    1.588247e-02               6  ...          0.945055          0.945055        0.947253       0.031391               36
11       0.216221      0.016356         0.007181    3.989698e-04               6  ...          0.945055          0.934066        0.947253       0.032151               36
12       0.213030      0.013727         0.020744    1.607398e-02               8  ...          0.956044          0.956044        0.956044       0.025059               16
13       0.215424      0.015553         0.007383    4.863050e-04               8  ...          0.945055          0.967033        0.960440       0.022628                5
14       0.216222      0.016658         0.020146    1.572374e-02               8  ...          0.945055          0.956044        0.951648       0.028317               32
15       0.201262      0.002918         0.020944    1.589495e-02               8  ...          0.934066          0.912088        0.938462       0.034471               44
16       0.216421      0.015515         0.007779    3.989697e-04              10  ...          0.934066          0.967033        0.956044       0.026917               16
17       0.202459      0.001262         0.020545    1.580184e-02              10  ...          0.945055          0.956044        0.956044       0.025059               16
18       0.216222      0.017801         0.021742    1.808903e-02              10  ...          0.956044          0.967033        0.958242       0.025441               11
19       0.205850      0.003816         0.020545    1.623665e-02              10  ...          0.934066          0.956044        0.947253       0.032151               36
20       0.215642      0.018485         0.008160    4.094884e-04              12  ...          0.945055          0.956044        0.953846       0.021308               27
21       0.233776      0.017645         0.013962    1.297092e-02              12  ...          0.956044          0.956044        0.953846       0.024474               27
22       0.211633      0.014500         0.020545    1.623669e-02              12  ...          0.945055          0.945055        0.947253       0.028991               36
23       0.208442      0.013050         0.027925    1.725122e-02              12  ...          0.945055          0.945055        0.947253       0.031391               36
24       0.214027      0.014014         0.006982    7.136645e-07             NaN  ...          0.956044          0.967033        0.960440       0.022628                5
25       0.226220      0.013665         0.007156    4.138263e-04             NaN  ...          0.945055          0.967033        0.956044       0.025059               16
26       0.235570      0.013883         0.015558    1.567553e-02             NaN  ...          0.945055          0.967033        0.960440       0.022628                5
27       0.206066      0.003024         0.013946    1.343517e-02             NaN  ...          0.945055          0.967033        0.958242       0.023466               11
28       0.210238      0.015313         0.015359    9.864945e-03             NaN  ...          0.945055          0.956044        0.949451       0.024670               35
29       0.220011      0.017736         0.007380    4.887529e-04             NaN  ...          0.945055          0.956044        0.953846       0.025441               27
30       0.219613      0.018187         0.013963    1.297294e-02             NaN  ...          0.956044          0.945055        0.956044       0.025059               16
31       0.213229      0.013010         0.014561    1.267221e-02             NaN  ...          0.945055          0.956044        0.951648       0.024670               32
32       0.229786      0.029737         0.007380    4.886166e-04             NaN  ...          0.956044          0.956044        0.947253       0.028991               36
33       0.211642      0.014089         0.014752    1.408654e-02             NaN  ...          0.945055          0.956044        0.956044       0.025059               16
34       0.207644      0.012221         0.015160    1.436160e-02             NaN  ...          0.945055          0.956044        0.956044       0.025059               16
35       0.223801      0.015001         0.015160    1.487844e-02             NaN  ...          0.934066          0.956044        0.953846       0.026374               27
36       0.211833      0.010509         0.013963    1.247279e-02             NaN  ...          0.956044          0.945055        0.951648       0.028317               32
37       0.208841      0.012875         0.013963    1.297299e-02             NaN  ...          0.956044          0.956044        0.956044       0.027800               16
38       0.226594      0.017845         0.014162    1.337163e-02             NaN  ...          0.945055          0.945055        0.947253       0.031391               36
39       0.216222      0.010841         0.020744    1.604923e-02             NaN  ...          0.912088          0.956044        0.945055       0.035438               43
40       0.223004      0.015238         0.020545    1.540656e-02             NaN  ...          0.956044          0.967033        0.958242       0.025441               11
41       0.222006      0.017847         0.007779    3.989458e-04             NaN  ...          0.956044          0.956044        0.956044       0.025059               16
42       0.212232      0.012669         0.014162    1.337142e-02             NaN  ...          0.967033          0.978022        0.969231       0.018906                1
43       0.186702      0.007888         0.008178    1.162500e-03             NaN  ...          0.956044          0.967033        0.962637       0.021534                4

[44 rows x 17 columns]
                                               params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.960440                5           0.945055
1               {'max_depth': 6, 'n_estimators': 200}         0.958242               11           0.923077
2               {'max_depth': 8, 'n_estimators': 100}         0.956044               16           0.934066
3               {'max_depth': 8, 'n_estimators': 200}         0.960440                5           0.934066
4              {'max_depth': 10, 'n_estimators': 100}         0.969231                1           0.945055
5              {'max_depth': 10, 'n_estimators': 200}         0.960440                5           0.934066
6              {'max_depth': 12, 'n_estimators': 100}         0.967033                3           0.934066
7              {'max_depth': 12, 'n_estimators': 200}         0.958242               11           0.934066
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.956044               16           0.923077
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.953846               27           0.923077
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.947253               36           0.901099
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.947253               36           0.901099
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.956044               16           0.923077
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.960440                5           0.934066
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.951648               32           0.912088
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.938462               44           0.901099
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.956044               16           0.923077
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.956044               16           0.923077
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.958242               11           0.923077
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.947253               36           0.901099
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.953846               27           0.923077
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.953846               27           0.912088
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.947253               36           0.912088
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.947253               36           0.901099
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.960440                5           0.934066
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.956044               16           0.934066
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.960440                5           0.934066
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.958242               11           0.934066
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.949451               35           0.912088
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.953846               27           0.923077
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.956044               16           0.923077
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.951648               32           0.912088
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.947253               36           0.901099
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.956044               16           0.923077
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.956044               16           0.923077
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.953846               27           0.923077
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.951648               32           0.912088
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.956044               16           0.912088
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.947253               36           0.901099
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.945055               43           0.901099
40                           {'min_samples_split': 2}         0.958242               11           0.923077
41                           {'min_samples_split': 3}         0.956044               16           0.923077
42                           {'min_samples_split': 5}         0.969231                1           0.945055
43                          {'min_samples_split': 10}         0.962637                4           0.934066
'''