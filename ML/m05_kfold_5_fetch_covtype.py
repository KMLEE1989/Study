import numpy as np
from sklearn.datasets import fetch_covtype
from sklearn.svm import SVC
import numpy as np
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

datasets = fetch_covtype()

x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold

x_train, x_test, y_train, y_test = train_test_split(x,y, shuffle=True, random_state=66, train_size=0.8)

n_splits=5
kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=66)

model1 = SVC()
#model2 = Perceptron()
#model3 = LinearSVC() 
#model4 = KNeighborsClassifier()
#model5 = LogisticRegression()
#model6 = DecisionTreeClassifier()
#model7 = RandomForestClassifier()

model1.fit(x_train, y_train)

result = model1.score(x_test, y_test) 

scores = cross_val_score(model1, x_train, y_train, cv=kfold)
print("SVC ACC: ", scores, "\n cross_val_score : ", round(np.mean(scores),4))


