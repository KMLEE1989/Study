import numpy as np
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.utils import all_estimators
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

datasets = load_breast_cancer()

x=datasets.data
y=datasets.target

from sklearn.model_selection import StratifiedKFold, train_test_split, KFold, cross_val_score

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

# ACC :  [0.92307692 0.92307692 0.9010989  0.91208791 0.91208791] 
#  cross_val_score :  0.9143

# SVC ACC :  [0.92307692 0.92307692 0.9010989  0.91208791 0.91208791] 
#  cross_val_score :  0.9143

# Perceptron ACC :  [0.91208791 0.82417582 0.84615385 0.81318681 0.76923077] 
#  cross_val_score :  0.833

# LinearSVC ACC :  [0.91208791 0.94505495 0.92307692 0.93406593 0.9010989 ] 
#  cross_val_score :  0.9231

# KNeighborsClassifier ACC:  [0.94505495 0.92307692 0.94505495 0.94505495 0.91208791] 
#  cross_val_score :  0.9341

# LogisticRegression ACC:  [0.94505495 0.91208791 0.95604396 0.95604396 0.94505495] 
#  cross_val_score :  0.9429

# DecisionTreeClassifier ACC:  [0.9010989  0.93406593 0.92307692 0.9010989  0.89010989] 
#  cross_val_score :  0.9099

# RandomForestClassifier ACC:  [0.95604396 0.97802198 1.         0.93406593 0.94505495] 
#  cross_val_score :  0.9626
