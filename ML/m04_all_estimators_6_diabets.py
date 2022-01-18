from sklearn import datasets
from sklearn.utils import all_estimators  #회기모델regressor은 r2스코어 accuracy= classifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
import warnings 
warnings.filterwarnings('ignore')

datasets=load_diabetes()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

allAlgorithm = all_estimators(type_filter='classifier')
#allAlgorithms = all_estimators(type_filter='regressor')


print("allAlgorithms : ", allAlgorithm)
print("모델의 갯수: ", len(allAlgorithm)) #3대장이 없어

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률 : ', acc)

    except:
        continue

# AdaBoostClassifier 의 정답률 :  0.0
# BaggingClassifier 의 정답률 :  0.0
# BernoulliNB 의 정답률 :  0.0
# CalibratedClassifierCV 의 정답률 :  0.0
# CategoricalNB 의 정답률 :  0.0
# ComplementNB 의 정답률 :  0.02247191011235955
# DecisionTreeClassifier 의 정답률 :  0.0
# DummyClassifier 의 정답률 :  0.0
# ExtraTreeClassifier 의 정답률 :  0.011235955056179775
# ExtraTreesClassifier 의 정답률 :  0.011235955056179775
# GaussianNB 의 정답률 :  0.011235955056179775
# GaussianProcessClassifier 의 정답률 :  0.011235955056179775
# GradientBoostingClassifier 의 정답률 :  0.011235955056179775
# HistGradientBoostingClassifier 의 정답률 :  0.02247191011235955
# KNeighborsClassifier 의 정답률 :  0.0
# LabelPropagation 의 정답률 :  0.011235955056179775
# LabelSpreading 의 정답률 :  0.011235955056179775
# LinearDiscriminantAnalysis 의 정답률 :  0.0
# LinearSVC 의 정답률 :  0.0
# LogisticRegression 의 정답률 :  0.0
# MLPClassifier 의 정답률 :  0.011235955056179775
# MultinomialNB 의 정답률 :  0.0
# NearestCentroid 의 정답률 :  0.011235955056179775
# PassiveAggressiveClassifier 의 정답률 :  0.0
# Perceptron 의 정답률 :  0.0
# RadiusNeighborsClassifier 의 정답률 :  0.0
# RandomForestClassifier 의 정답률 :  0.011235955056179775
# RidgeClassifier 의 정답률 :  0.0
# RidgeClassifierCV 의 정답률 :  0.0
# SGDClassifier 의 정답률 :  0.0
# SVC 의 정답률 :  0.011235955056179775