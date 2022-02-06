from tkinter.tix import LabelEntry
from catboost import train
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from tqdm import trange
from xgboost import XGBClassifier, XGBRegressor
import matplotlib as plt
from matplotlib.pyplot import boxplot
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures

path = "../_data/yoondata/"    

datasets = pd.read_csv(path + 'winequality-white.csv', sep=';',index_col=None, header=0)

datasets = datasets.values

x = datasets[:, :11]
y = datasets[:, 11]
# print(y.shape) #(4898,)

print("라벨: ", np.unique(y, return_counts=True))
# 라벨:  (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

newlist = []
for i in y:
    #print(i)
    if i<=4 :
        newlist += [0]    
    elif i<=7:
        newlist += [1]
    else:
        newlist += [2]
            
y=np.array(newlist)  

print(np.unique(y, return_counts=True))       
        
x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8, stratify = y)
#print(x_train.shape, y_train.shape)

scaler = PolynomialFeatures()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=1000,
    learning_rate = 0.1,
    max_depth = 6,
    min_child_weight = 1,
    subsample =1,
    colsample_bytree =1,
    reg_alpha =1,              
    reg_lambda=0,              
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor')

#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',          
          early_stopping_rounds=20)
end = time.time()

print( "걸린시간 :", end - start)

#4. 평가
results = model.score(x_test, y_test) 
print("results :", results)
y_pred = model.predict(x_test)

acc = accuracy_score(y_test, y_pred)
print("acc :", acc)
print('f1_score :', f1_score(y_test, y_pred, average='micro'))

''' 
걸린시간 : 5.479347467422485
results : 0.9377551020408164
acc : 0.9377551020408164
f1_score : 0.9377551020408164
'''