import numpy as np
from sklearn.datasets import load_wine
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = load_wine()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#model1 = SVC()
#model2 = Perceptron()
#model3 = LinearSVC() 
#model4 = KNeighborsClassifier()
#model5 = LogisticRegression()
#model6 = DecisionTreeClassifier()
model7 = RandomForestClassifier()

model7.fit(x_train, y_train)

result = model7.score(x_test, y_test) 

scores = cross_val_score(model7, x_train, y_train, cv=kfold)
print("RandomForestClassifier ACC: ", scores, "\n cross_val_score : ", round(np.mean(scores),4))

# SVC ACC:  [0.62068966 0.62068966 0.82142857 0.67857143 0.71428571] 
#  cross_val_score :  0.6911

# Perceptron ACC:  [0.68965517 0.27586207 0.39285714 0.60714286 0.5       ] 
#  cross_val_score :  0.4931

# LinearSVC ACC:  [0.65517241 0.89655172 0.96428571 0.67857143 0.64285714] 
#  cross_val_score :  0.7675

# KNeighborsClassifier ACC:  [0.62068966 0.65517241 0.67857143 0.75       0.75      ] 
#  cross_val_score :  0.6909

# LogisticRegression ACC:  [0.93103448 0.89655172 0.96428571 0.92857143 0.96428571]
#  cross_val_score :  0.9369

# DecisionTreeClassifier ACC:  [0.72413793 0.93103448 0.96428571 0.96428571 0.92857143] 
#  cross_val_score :  0.9025

# RandomForestClassifier ACC:  [0.96551724 0.96551724 1.         1.         0.92857143] 
#  cross_val_score :  0.9719

