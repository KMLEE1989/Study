from sklearn import datasets
from sklearn.utils import all_estimators  #회기모델regressor은 r2스코어 accuracy= classifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_covtype
import warnings 
warnings.filterwarnings('ignore')

datasets=fetch_covtype()
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
        print(name, '은 에러 터진 놈!!!!')
    
    
    
