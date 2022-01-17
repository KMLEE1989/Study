from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score
import pandas as pd

path = "../_data/kaggle/bike/"

train = pd.read_csv(path+'train.csv')
test_file = pd.read_csv(path+'test.csv')
submit_file = pd.read_csv(path+'sampleSubmission.csv')

x=train.drop(['datetime','casual','registered','count'],axis=1)
y=train['count']

test_file=test_file.drop(['datetime'], axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#model1 = Perceptron()
#model2 = LinearSVC() 
#model3 = SVC()
#model4 =  KNeighborsRegressor()
#model5 = LogisticRegression()
#model6 = DecisionTreeClassifier()
#model7 = RandomForestClassifier()
#model8=LinearRegression()
model1=KNeighborsRegressor()
model2=LinearRegression()
model3=DecisionTreeRegressor()
model4=RandomForestRegressor()

model4.fit(x_train, y_train)

result=model4.score(x_test, y_test)

from sklearn.metrics import r2_score
y_predict = model4.predict(x_test)
r2= r2_score(y_test, y_predict)

print("RandomForestRegressor: ", result)
print("r2_score: ", r2)

# Perceptron:  0.0009182736455463728
# r2_score:  -0.8090973983645309

# LinearSVC:  0.0013774104683195593
# r2_score:  -1.841089975379857

# SVC:  0.018365472910927456
# r2_score:  -0.753971764735198

# KNeighborsRegressor:  0.1867313357142426
# r2_score:  0.1867313357142426

# LinearRegression:  0.2494896826312223
# r2_score:  0.2494896826312223

# DecisionTreeRegressor:  -0.20431747361632024
# r2_score:  -0.20431747361632024

