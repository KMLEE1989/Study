from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_wine()
x = datasets.data
y = datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
#print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41


#allAlgorithms = all_estimators(type_filter = 'regressor')  # regressor에 대한 모든 측정기

#print("allAlgorithms: ", allAlgorithms)  # [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수: 55

for (name, algorithm) in allAlgorithms:   # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),...
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        n_splits = 5
        kfold = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 66)
        scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다. 
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률: ', acc, round(np.mean(scores),4))
    except:                     
        #continue   
        print(name, '은 에러')   

'''
AdaBoostClassifier 의 정답률:  0.9473684210526315 0.9429
BaggingClassifier 의 정답률:  0.9385964912280702 0.9297
BernoulliNB 의 정답률:  0.6403508771929824 0.611
CalibratedClassifierCV 의 정답률:  0.9649122807017544 0.9648
CategoricalNB 은 에러
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.7807017543859649 0.8527
DecisionTreeClassifier 의 정답률:  0.9298245614035088 0.9143
DummyClassifier 의 정답률:  0.6403508771929824 0.6242
ExtraTreeClassifier 의 정답률:  0.8859649122807017 0.9187
ExtraTreesClassifier 의 정답률:  0.9649122807017544 0.967
GaussianNB 의 정답률:  0.9210526315789473 0.9363
GaussianProcessClassifier 의 정답률:  0.9649122807017544 0.9604
GradientBoostingClassifier 의 정답률:  0.9473684210526315 0.9429
HistGradientBoostingClassifier 의 정답률:  0.9736842105263158 0.9582
KNeighborsClassifier 의 정답률:  0.956140350877193 0.9758
LabelPropagation 의 정답률:  0.9473684210526315 0.9736
LabelSpreading 의 정답률:  0.9473684210526315 0.9692
LinearDiscriminantAnalysis 의 정답률:  0.9473684210526315 0.9582
LinearSVC 의 정답률:  0.9736842105263158 0.967
LogisticRegression 의 정답률:  0.9649122807017544 0.9626
LogisticRegressionCV 의 정답률:  0.9736842105263158 0.967
MLPClassifier 의 정답률:  0.9736842105263158 0.9626
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.8508771929824561 0.8505
NearestCentroid 의 정답률:  0.9298245614035088 0.9341
NuSVC 의 정답률:  0.9473684210526315 0.9473
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9473684210526315 0.9407
QuadraticDiscriminantAnalysis 의 정답률:  0.9385964912280702 0.9604
RadiusNeighborsClassifier 은 에러
RandomForestClassifier 의 정답률:  0.9736842105263158 0.9538
RidgeClassifier 의 정답률:  0.9473684210526315 0.956
RidgeClassifierCV 의 정답률:  0.9473684210526315 0.9604
SGDClassifier 의 정답률:  0.9912280701754386 0.9714
SVC 의 정답률:  0.9736842105263158 0.9736
StackingClassifier 은 에러
VotingClassifier 은 에러
PS D:\Study>  d:; cd 'd:\Study'; & 'C:\ProgramData\Anaconda3\envs\tf270gpu\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '51578' '--' 'd:\Study\ml\m06_kfold_all)estimator_3_wine.py' 
AdaBoostClassifier 의 정답률:  0.8888888888888888 0.8096
BaggingClassifier 의 정답률:  1.0 0.965
BernoulliNB 의 정답률:  0.4166666666666667 0.352
CalibratedClassifierCV 의 정답률:  0.9722222222222222 0.986
CategoricalNB 의 정답률:  0.5 nan
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.8611111111111112 0.8879
DecisionTreeClassifier 의 정답률:  0.9444444444444444 0.9293
DummyClassifier 의 정답률:  0.4166666666666667 0.3943
ExtraTreeClassifier 의 정답률:  0.8611111111111112 0.8596
ExtraTreesClassifier 의 정답률:  1.0 0.9719
GaussianNB 의 정답률:  1.0 0.965
GaussianProcessClassifier 의 정답률:  1.0 0.965
GradientBoostingClassifier 의 정답률:  0.9722222222222222 0.9224
HistGradientBoostingClassifier 의 정답률:  0.9722222222222222 0.9579
KNeighborsClassifier 의 정답률:  1.0 0.9579
LabelPropagation 의 정답률:  1.0 0.9433
LabelSpreading 의 정답률:  1.0 0.9433
LinearDiscriminantAnalysis 의 정답률:  1.0 0.9648
LinearSVC 의 정답률:  0.9722222222222222 0.986
LogisticRegression 의 정답률:  1.0 0.9719
LogisticRegressionCV 의 정답률:  0.9722222222222222 0.9788
MLPClassifier 의 정답률:  0.9722222222222222 0.9653
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.9444444444444444 0.9438
NearestCentroid 의 정답률:  1.0 0.9441
NuSVC 의 정답률:  1.0 0.9648
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9722222222222222 0.9862
Perceptron 의 정답률:  0.9722222222222222 0.9931
QuadraticDiscriminantAnalysis 의 정답률:  0.9722222222222222 0.9791
RadiusNeighborsClassifier 의 정답률:  0.9722222222222222 0.9507
RandomForestClassifier 의 정답률:  1.0 0.9788
RidgeClassifier 의 정답률:  1.0 0.9791
RidgeClassifierCV 의 정답률:  0.9722222222222222 0.9719
SGDClassifier 의 정답률:  1.0 0.9791
SVC 의 정답률:  1.0 0.9719
StackingClassifier 은 에러
VotingClassifier 은 에러
'''