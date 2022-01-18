import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

path = "../_data/kaggle/bike/"

train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+'sampleSubmission.csv')

x=train.drop(['datetime','casual','registered','count'],axis=1)
y=train['count']

test_file=test_file.drop(['datetime'], axis=1)

from sklearn.model_selection import KFold, cross_val_score, train_test_split, StratifiedKFold
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

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

# KNeighborsRegressor R2 :  [0.24273368 0.20408445 0.18862976 0.20000081 0.20534716] 
# cross_val_score :  0.2082

# LogisticRegression R2 :  [0.0206659  0.0195178  0.01435132 0.01780586 0.01952901] 
#  cross_val_score :  0.0184

# LinearRegression R2 :  [0.27878294 0.25028934 0.24072136 0.2644018  0.27369838] 
#  cross_val_score :  0.2616

# DecisionTreeRegressor R2 :  [-0.14211031 -0.16544906 -0.29979115 -0.1561347  -0.18198433] 
#  cross_val_score :  -0.1891

# RandomForestRegressor R2 :  [0.36246797 0.27688007 0.22117844 0.28356551 0.30452593] 
#  cross_val_score :  0.2897