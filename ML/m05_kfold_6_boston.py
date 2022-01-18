import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score
from sklearn.svm import SVC
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


datasets = load_boston()

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

# KNeighborsRegressor R2 :  [0.38689566 0.52994483 0.3434155  0.55325748 0.51995804] 
#  cross_val_score :  0.4667

#LogisticRegressionX

# LinearRegression R2 :  [0.5815212  0.69885237 0.6537276  0.77449543 0.70223459] 
#  cross_val_score :  0.6822

# DecisionTreeRegressor R2 :  [0.78964196 0.62695345 0.63103974 0.69429893 0.78130752] 
#  cross_val_score :  0.7046

# RandomForestRegressor R2 :  [0.87971361 0.72877898 0.79993362 0.86422418 0.8897851 ] 
#  cross_val_score :  0.8325






