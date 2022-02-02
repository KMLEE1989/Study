from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time

import pickle

path ='../_save/'
datasets = pickle.load(open(path + 'm26_pickle1_save_datasets.dat', 'rb'))

x=datasets.data
y=datasets['target']


x_train, x_test,y_train,y_test= train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)


model=XGBClassifier(n_estimators=1000,
    learning_rate=0.025,               
    max_depth=4,
    min_child_weight = 10,
    subsample=0.5,
    colsample_bytree = 1, 
    reg_alpha = 1,              
    reg_lambda=0, 
    tree_method='gpu_hist',
    predictor='gpu_predictor',
    gpu_id=0,
)

start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          eval_metric='mlogloss',
          early_stopping_rounds=10) #rmse, mae, logloss, error
end = time.time()

print("걸린시간 : ", end - start)

results=model.score(x_test, y_test)

print("results : ", round(results,5))

y_predict = model.predict(x_test)

acc= accuracy_score(y_test, y_predict)

print("acc : ", round(acc,4))


                        

