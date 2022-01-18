import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import load_diabetes
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

datasets = load_diabetes()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

#model1 = KNeighborsRegressor()
#model2 = LogisticRegression()
#model3 = LinearRegression()
#model4 = DecisionTreeRegressor()
model5 = RandomForestRegressor()

model5.fit(x_train, y_train)

result = model5.score(x_test, y_test) 

scores = cross_val_score(model5, x_train, y_train, cv=kfold)
print("RandomForestRegressor R2 : ", scores, "\n cross_val_score : ", round(np.mean(scores),4))

# KNeighborsRegressor R2 :  [0.39650197 0.35421275 0.30652706 0.47874019 0.47266077] 
#  cross_val_score :  0.4017

# LogisticRegression R2 :  [0.01408451 0.01408451 0.01408451 0.01428571 0.01428571] 
#  cross_val_score :  0.0142

# LinearRegression R2 :  [0.48004831 0.53238253 0.44089734 0.46431556 0.52245017] 
#  cross_val_score :  0.488

# DecisionTreeRegressor R2 :  [-0.24557395 -0.06495858 -0.03205771  0.12382916 -0.01199486] 
#  cross_val_score :  -0.0462

# RandomForestRegressor R2 :  [0.32429365 0.51699583 0.46703249 0.57806056 0.4484989 ] 
#  cross_val_score :  0.467
