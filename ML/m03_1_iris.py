import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

#print(x.shape, y.shape)  #(150, 4) (150,)
#print(y)
#print(np.unique(y)) #[0 1 2]  라벨값이란?  여기서는 4래 여기서 (150,4) 그리고 (150,3) 으로 만들자! 원핫 인코딩을 이용해!
# y=to_categorical(y)
# print(y)
# print(y.shape)  #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


model1= Perceptron()
model2=LinearSVC()
model3=SVC()
model4=KNeighborsClassifier()
model5=LogisticRegression()
model6=DecisionTreeClassifier()
model7=RandomForestClassifier()

model7.fit(x_train, y_train)

result = model7.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model7.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("RandomForestClassifier : ", result)
print("")
#print("accuracy_score: ", acc)

# result:  0.9666666666666667
# accuracy_score:  0.9666666666666667

# Perceptron :  0.9333333333333333
# LinearSVC :  0.9666666666666667
# SVC :  0.9666666666666667
# KNeighborsClassifier:  0.9666666666666667
# LogisticRegression :  1.0
# DecisionTreeClassifier :  0.9666666666666667
# RandomForestClassifier :  0.9
