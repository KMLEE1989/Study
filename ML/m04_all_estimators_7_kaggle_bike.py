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
from sklearn.metrics import accuracy_score, r2_score
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

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#allAlgorithm = all_estimators(type_filter='classifier')
allAlgorithm = all_estimators(type_filter='regressor')


print("allAlgorithms : ", allAlgorithm)
print("모델의 갯수: ", len(allAlgorithm)) #3대장이 없어

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)

    except:
        continue
    
