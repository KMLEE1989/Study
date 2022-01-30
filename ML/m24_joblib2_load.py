from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
# import warnings
# warnings.filterwarnings('ignore')

#datasets = fetch_california_housing()
datasets = load_boston()
x=datasets.data
y=datasets['target']

# print(x.shape, y.shape) (20640, 8) (20640,)

x_train, x_test,y_train,y_test= train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

#import pickle
path = 'D:\save'
#model = pickle.load(open(path + 'm23_pickle1_save.dat', 'rb'))
import joblib
model = joblib.load(path + 'm24_joblib1_save.dat')

result = model.score(x_test, y_test)
print("results : ", result)

y_predict = model.predict(x_test)
r2=r2_score(y_test, y_predict)
print("r2: ", r2)


print("===================================")
hist = model.evals_result()
print(hist)


