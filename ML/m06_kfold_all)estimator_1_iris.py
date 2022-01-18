from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
datasets = load_iris()
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
        print(name, '의 정답률: ', acc, scores, round(np.mean(scores),4))
    except:                     # 에러나는 것 빼고 계속해라.
        #continue   
        print(name, '은 에러')

'''
AdaBoostClassifier 의 정답률:  0.6333333333333333 [0.91666667 0.70833333 0.95833333 0.95833333 0.95833333] 0.9
BaggingClassifier 의 정답률:  0.9333333333333333 [0.875      1.         0.95833333 0.95833333 0.91666667] 0.9417
BernoulliNB 의 정답률:  0.4 [0.29166667 0.375      0.375      0.375      0.45833333] 0.375
CalibratedClassifierCV 의 정답률:  0.9666666666666667 [0.875      0.95833333 0.79166667 0.875      0.91666667] 0.8833
CategoricalNB 의 정답률:  0.3333333333333333 [       nan        nan 0.33333333 0.375             nan] nan
ClassifierChain 은 에러
ComplementNB 의 정답률:  0.6666666666666666 [0.66666667 0.66666667 0.66666667 0.66666667 0.66666667] 0.6667
DecisionTreeClassifier 의 정답률:  0.9666666666666667 [0.875      1.         0.95833333 0.95833333 0.91666667] 0.9417
DummyClassifier 의 정답률:  0.3 [0.375      0.33333333 0.33333333 0.33333333 0.33333333] 0.3417
ExtraTreeClassifier 의 정답률:  0.9333333333333333 [0.91666667 1.         0.83333333 0.91666667 0.91666667] 0.9167
ExtraTreesClassifier 의 정답률:  0.9666666666666667 [0.91666667 1.         0.91666667 0.95833333 0.91666667] 0.9417
GaussianNB 의 정답률:  0.9666666666666667 [0.91666667 1.         0.91666667 0.95833333 0.91666667] 0.9417
GaussianProcessClassifier 의 정답률:  0.9666666666666667 [0.91666667 1.         0.83333333 0.875      0.95833333] 0.9167
GradientBoostingClassifier 의 정답률:  0.9666666666666667 [0.875      1.         0.95833333 0.95833333 0.91666667] 0.9417
HistGradientBoostingClassifier 의 정답률:  0.8666666666666667 [0.875      1.         0.91666667 0.95833333 0.91666667] 0.9333
KNeighborsClassifier 의 정답률:  1.0 [0.95833333 1.         0.95833333 0.95833333 0.95833333] 0.9667
LabelPropagation 의 정답률:  0.9666666666666667 [0.91666667 1.         0.91666667 0.95833333 0.91666667] 0.9417
LabelSpreading 의 정답률:  0.9666666666666667 [0.91666667 1.         0.91666667 0.95833333 0.91666667] 0.9417
LinearDiscriminantAnalysis 의 정답률:  1.0 [1.         1.         0.95833333 1.         0.91666667] 0.975
LinearSVC 의 정답률:  0.9666666666666667 [0.875      1.         0.83333333 0.875      0.875     ] 0.8917
LogisticRegression 의 정답률:  0.9666666666666667 [0.91666667 1.         0.83333333 0.875      0.95833333] 0.9167
LogisticRegressionCV 의 정답률:  1.0 [0.91666667 1.         0.95833333 1.         0.91666667] 0.9583
MLPClassifier 의 정답률:  0.9333333333333333 [0.91666667 1.         0.875      0.91666667 0.91666667] 0.925
MultiOutputClassifier 은 에러
MultinomialNB 의 정답률:  0.6333333333333333 [0.75       0.70833333 0.66666667 0.66666667 0.66666667] 0.6917
NearestCentroid 의 정답률:  0.9666666666666667 [0.875      1.         0.875      0.875      0.95833333] 0.9167
NuSVC 의 정답률:  0.9666666666666667 [0.91666667 1.         0.95833333 0.95833333 0.91666667] 0.95
OneVsOneClassifier 은 에러
OneVsRestClassifier 은 에러
OutputCodeClassifier 은 에러
PassiveAggressiveClassifier 의 정답률:  0.9 [0.875      1.         0.91666667 0.91666667 0.875     ] 0.9167
Perceptron 의 정답률:  0.9333333333333333 [0.79166667 0.75       0.79166667 0.70833333 0.91666667] 0.7917
QuadraticDiscriminantAnalysis 의 정답률:  1.0 [1.         1.         0.91666667 1.         0.91666667] 0.9667
RadiusNeighborsClassifier 의 정답률:  0.4666666666666667 [0.54166667 0.375      0.5        0.5        0.41666667] 0.4667
RandomForestClassifier 의 정답률:  0.9333333333333333 [0.91666667 1.         0.95833333 0.95833333 0.91666667] 0.95
RidgeClassifier 의 정답률:  0.9333333333333333 [0.83333333 0.91666667 0.79166667 0.83333333 0.83333333] 0.8417
RidgeClassifierCV 의 정답률:  0.8333333333333334 [0.83333333 0.91666667 0.79166667 0.79166667 0.79166667] 0.825
SGDClassifier 의 정답률:  0.8333333333333334 [0.875      1.         0.75       0.91666667 0.91666667] 0.8917
SVC 의 정답률:  1.0 [0.95833333 1.         0.95833333 0.95833333 0.91666667] 0.9583
StackingClassifier 은 에러
VotingClassifier 은 에러
'''
