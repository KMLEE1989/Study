import numpy as np, pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

#1) 데이터
datasets = load_wine()
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
Fitting 5 folds for each of 56 candidates, totalling 280 fits
최적의 매개변수:  RandomForestClassifier(max_depth=6, n_estimators=200)
최적의 파라미터:  {'max_depth': 6, 'n_estimators': 200}
best_score_ :  0.9788177339901478
model.score:  1.0
accuracy_score:  1.0
최적 튠 ACC:  1.0
걸린 시간:  9.266215324401855
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
0        0.159772      0.005897         0.014960    1.396277e-02               6  ...          0.964286          0.964286        0.964532       0.022593               15
1        0.325928      0.021872         0.022739    1.709374e-02               6  ...          1.000000          0.964286        0.978818       0.017301                1
2        0.149201      0.017725         0.036702    1.690641e-02               8  ...          1.000000          0.964286        0.971675       0.026665                7
3        0.370609      0.015212         0.033710    1.614824e-02               8  ...          0.964286          0.964286        0.971675       0.014171                7
4        0.176727      0.014945         0.007181    3.989221e-04              10  ...          0.928571          0.964286        0.964532       0.022593               15
5        0.345077      0.016605         0.033510    1.558655e-02              10  ...          0.964286          0.964286        0.971675       0.014171                7
6        0.180118      0.015173         0.008378    1.017130e-03              12  ...          0.928571          0.964286        0.957389       0.026796               29
7        0.361035      0.013485         0.028125    1.694222e-02              12  ...          0.964286          0.964286        0.971675       0.014171                7
8        0.191519      0.007847         0.006950    6.175161e-05               6  ...          0.928571          0.964286        0.964532       0.022593               15
9        0.174334      0.017934         0.020544    1.671912e-02               6  ...          0.892857          0.964286        0.943350       0.036241               52
10       0.168948      0.012751         0.013763    1.308627e-02               6  ...          0.892857          0.928571        0.943103       0.036570               54
11       0.168550      0.017650         0.014362    1.476052e-02               6  ...          0.928571          0.964286        0.957389       0.026796               29
12       0.181714      0.015673         0.021143    1.660970e-02               8  ...          0.964286          0.964286        0.964532       0.022593               15
13       0.174137      0.015738         0.020143    1.572008e-02               8  ...          0.928571          0.964286        0.950493       0.017904               45
14       0.179320      0.013402         0.007181    3.985883e-04               8  ...          0.964286          0.964286        0.964532       0.022593               15
15       0.169945      0.017154         0.020944    1.714729e-02               8  ...          0.964286          0.964286        0.964532       0.022593               15
16       0.173536      0.019472         0.007380    4.885582e-04              10  ...          0.928571          0.964286        0.957389       0.026796               29
17       0.178922      0.024156         0.023738    1.930052e-02              10  ...          0.964286          0.964286        0.964778       0.000603               14
18       0.180916      0.018782         0.020944    1.711227e-02              10  ...          0.964286          0.928571        0.950246       0.036521               46
19       0.171938      0.017880         0.020545    1.709599e-02              10  ...          0.964286          0.964286        0.957635       0.014542               27
20       0.171142      0.021275         0.022141    1.849108e-02              12  ...          0.928571          0.964286        0.971429       0.026726               13
21       0.186900      0.012656         0.007181    3.988981e-04              12  ...          0.892857          0.964286        0.943350       0.028945               48
22       0.136235      0.028093         0.013763    1.307090e-02              12  ...          0.964286          0.964286        0.957389       0.035046               35
23       0.132645      0.027545         0.012168    9.881162e-03              12  ...          0.964286          0.964286        0.950739       0.017118               44
24       0.146807      0.032687         0.007182    3.981219e-04             NaN  ...          0.892857          0.964286        0.957389       0.035046               35
25       0.140424      0.030833         0.006981    9.536743e-08             NaN  ...          0.892857          0.964286        0.957389       0.035046               35
26       0.136234      0.028881         0.012367    1.077108e-02             NaN  ...          0.928571          0.964286        0.957635       0.014542               27
27       0.131848      0.023881         0.013165    1.187445e-02             NaN  ...          0.928571          0.964286        0.964532       0.022593               15
28       0.140823      0.028970         0.011769    9.574699e-03             NaN  ...          0.892857          0.964286        0.943350       0.028945               48
29       0.136834      0.031154         0.013963    1.346978e-02             NaN  ...          0.928571          0.964286        0.957389       0.026796               29
30       0.136834      0.031723         0.013563    1.267173e-02             NaN  ...          0.892857          0.964286        0.957389       0.035046               35
31       0.135837      0.029882         0.012766    1.156921e-02             NaN  ...          0.892857          0.964286        0.957389       0.035046               35
32       0.139228      0.031308         0.013165    1.137838e-02             NaN  ...          0.928571          0.964286        0.957389       0.026796               29
33       0.138430      0.029903         0.009774    5.584908e-03             NaN  ...          0.928571          0.964286        0.957389       0.026796               29
34       0.134241      0.026609         0.017354    1.310397e-02             NaN  ...          0.892857          0.964286        0.922660       0.026106               56
35       0.150198      0.033273         0.007181    3.987079e-04             NaN  ...          0.892857          0.964286        0.943350       0.028945               48
36       0.127858      0.024409         0.018750    1.482508e-02             NaN  ...          0.964286          0.964286        0.957389       0.035046               35
37       0.131847      0.026675         0.007181    3.988266e-04             NaN  ...          0.892857          0.964286        0.943350       0.036241               52
38       0.142619      0.034009         0.006981    1.907349e-07             NaN  ...          0.857143          0.964286        0.936207       0.041991               55
39       0.155584      0.025992         0.007181    3.987074e-04             NaN  ...          0.892857          0.964286        0.943350       0.028945               48
40       0.151594      0.036235         0.013564    1.267177e-02               6  ...          0.928571          0.964286        0.964532       0.022593               15
41       0.151192      0.034843         0.007181    3.988028e-04               6  ...          1.000000          0.964286        0.978818       0.017301                1
42       0.137832      0.028701         0.011968    7.978892e-03               6  ...          0.892857          0.964286        0.950246       0.036521               46
43       0.143616      0.043404         0.015559    1.181082e-02               6  ...          0.964286          0.964286        0.964532       0.022593               15
44       0.124068      0.028442         0.008973    3.991345e-03               8  ...          0.964286          0.964286        0.971675       0.014171                7
45       0.139028      0.027894         0.012566    1.116974e-02               8  ...          0.892857          0.964286        0.957389       0.035046               35
46       0.135039      0.028950         0.019747    1.563515e-02               8  ...          1.000000          0.964286        0.978818       0.017301                1
47       0.132246      0.027787         0.011170    8.884554e-03               8  ...          1.000000          0.964286        0.978571       0.028571                5
48       0.125265      0.025642         0.013962    1.206765e-02              10  ...          0.928571          0.964286        0.964532       0.022593               15
49       0.148204      0.033936         0.007181    3.988506e-04              10  ...          1.000000          0.964286        0.978818       0.017301                1
50       0.140823      0.031598         0.006782    3.989937e-04              10  ...          0.892857          0.964286        0.957389       0.035046               35
51       0.140026      0.030189         0.011768    9.083971e-03              10  ...          0.928571          0.964286        0.964532       0.022593               15
52       0.144812      0.030632         0.007181    3.988744e-04              12  ...          0.892857          0.964286        0.957389       0.035046               35
53       0.116688      0.030526         0.013763    1.307026e-02              12  ...          0.928571          0.964286        0.964532       0.022593               15
54       0.150398      0.010896         0.012766    6.950054e-03              12  ...          0.964286          0.964286        0.971675       0.014171                7
55       0.118484      0.021339         0.009176    2.394160e-03              12  ...          1.000000          0.964286        0.971921       0.025943                6

[56 rows x 17 columns]
                                               params  mean_test_score  rank_test_score  split0_test_score
0               {'max_depth': 6, 'n_estimators': 100}         0.964532               15           0.965517
1               {'max_depth': 6, 'n_estimators': 200}         0.978818                1           0.965517
2               {'max_depth': 8, 'n_estimators': 100}         0.971675                7           0.965517
3               {'max_depth': 8, 'n_estimators': 200}         0.971675                7           0.965517
4              {'max_depth': 10, 'n_estimators': 100}         0.964532               15           0.965517
5              {'max_depth': 10, 'n_estimators': 200}         0.971675                7           0.965517
6              {'max_depth': 12, 'n_estimators': 100}         0.957389               29           0.965517
7              {'max_depth': 12, 'n_estimators': 200}         0.971675                7           0.965517
8             {'max_depth': 6, 'min_samples_leaf': 3}         0.964532               15           0.965517
9             {'max_depth': 6, 'min_samples_leaf': 5}         0.943350               52           1.000000
10            {'max_depth': 6, 'min_samples_leaf': 7}         0.943103               54           0.965517
11           {'max_depth': 6, 'min_samples_leaf': 10}         0.957389               29           1.000000
12            {'max_depth': 8, 'min_samples_leaf': 3}         0.964532               15           0.965517
13            {'max_depth': 8, 'min_samples_leaf': 5}         0.950493               45           0.965517
14            {'max_depth': 8, 'min_samples_leaf': 7}         0.964532               15           0.965517
15           {'max_depth': 8, 'min_samples_leaf': 10}         0.964532               15           0.965517
16           {'max_depth': 10, 'min_samples_leaf': 3}         0.957389               29           0.965517
17           {'max_depth': 10, 'min_samples_leaf': 5}         0.964778               14           0.965517
18           {'max_depth': 10, 'min_samples_leaf': 7}         0.950246               46           0.965517
19          {'max_depth': 10, 'min_samples_leaf': 10}         0.957635               27           0.965517
20           {'max_depth': 12, 'min_samples_leaf': 3}         0.971429               13           1.000000
21           {'max_depth': 12, 'min_samples_leaf': 5}         0.943350               48           0.965517
22           {'max_depth': 12, 'min_samples_leaf': 7}         0.957389               35           1.000000
23          {'max_depth': 12, 'min_samples_leaf': 10}         0.950739               44           0.965517
24    {'min_samples_leaf': 3, 'min_samples_split': 2}         0.957389               35           0.965517
25    {'min_samples_leaf': 3, 'min_samples_split': 3}         0.957389               35           0.965517
26    {'min_samples_leaf': 3, 'min_samples_split': 5}         0.957635               27           0.965517
27   {'min_samples_leaf': 3, 'min_samples_split': 10}         0.964532               15           1.000000
28    {'min_samples_leaf': 5, 'min_samples_split': 2}         0.943350               48           0.965517
29    {'min_samples_leaf': 5, 'min_samples_split': 3}         0.957389               29           0.965517
30    {'min_samples_leaf': 5, 'min_samples_split': 5}         0.957389               35           0.965517
31   {'min_samples_leaf': 5, 'min_samples_split': 10}         0.957389               35           0.965517
32    {'min_samples_leaf': 7, 'min_samples_split': 2}         0.957389               29           0.965517
33    {'min_samples_leaf': 7, 'min_samples_split': 3}         0.957389               29           0.965517
34    {'min_samples_leaf': 7, 'min_samples_split': 5}         0.922660               56           0.896552
35   {'min_samples_leaf': 7, 'min_samples_split': 10}         0.943350               48           0.965517
36   {'min_samples_leaf': 10, 'min_samples_split': 2}         0.957389               35           0.965517
37   {'min_samples_leaf': 10, 'min_samples_split': 3}         0.943350               52           0.931034
38   {'min_samples_leaf': 10, 'min_samples_split': 5}         0.936207               55           0.965517
39  {'min_samples_leaf': 10, 'min_samples_split': 10}         0.943350               48           0.965517
40           {'max_depth': 6, 'min_samples_split': 2}         0.964532               15           0.965517
41           {'max_depth': 6, 'min_samples_split': 3}         0.978818                1           0.965517
42           {'max_depth': 6, 'min_samples_split': 5}         0.950246               46           0.965517
43          {'max_depth': 6, 'min_samples_split': 10}         0.964532               15           0.965517
44           {'max_depth': 8, 'min_samples_split': 2}         0.971675                7           0.965517
45           {'max_depth': 8, 'min_samples_split': 3}         0.957389               35           0.965517
46           {'max_depth': 8, 'min_samples_split': 5}         0.978818                1           0.965517
47          {'max_depth': 8, 'min_samples_split': 10}         0.978571                5           1.000000
48          {'max_depth': 10, 'min_samples_split': 2}         0.964532               15           0.965517
49          {'max_depth': 10, 'min_samples_split': 3}         0.978818                1           0.965517
50          {'max_depth': 10, 'min_samples_split': 5}         0.957389               35           0.965517
51         {'max_depth': 10, 'min_samples_split': 10}         0.964532               15           0.965517
52          {'max_depth': 12, 'min_samples_split': 2}         0.957389               35           0.965517
53          {'max_depth': 12, 'min_samples_split': 3}         0.964532               15           0.965517
54          {'max_depth': 12, 'min_samples_split': 5}         0.971675                7           0.965517
55         {'max_depth': 12, 'min_samples_split': 10}         0.971921                6           0.931034
'''