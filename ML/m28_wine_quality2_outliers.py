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

path = "../_data/yoondata/"    

datasets = pd.read_csv(path + 'winequality-white.csv', sep=';',index_col=None, header=0)

def remove_outlier(input_data):
    q1 = input_data.quantile(0.25) # 제 1사분위수
    q3 = input_data.quantile(0.75) # 제 3사분위수
    iqr = q3 - q1 # IQR(Interquartile range) 계산
    minimum = q1 - (iqr * 1.5) # IQR 최솟값
    maximum = q3 + (iqr * 1.5) # IQR 최댓값
    ### IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치) ###
    df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
    return df_removed_outlier

# 이상치 제거한 데이터셋
datasets = remove_outlier(datasets)
#print(datasets[:40])

# 이상치 채워주기
datasets = datasets.interpolate()
#print(datasets[:40])
############################################################################################

'''
x = datasets.drop(['quality'], axis=1)
y = datasets['quality']
#print(x.shape, y.shape) #(4898, 11) (4898,)
x_train, x_test, y_train, y_test = train_test_split (x, y, shuffle=True, random_state=66, train_size=0.8, stratify = y)
print(x_train.shape, y_train.shape)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
#2. 모델
model = XGBClassifier(
    n_jobs = -1,
    n_estimators=10000,
    learning_rate = 0.4,
    max_depth = 6,
    min_child_weight = 0.9,
    subsample =1,
    colsample_bytree =0.9,
    reg_alpha =1,              
    reg_lambda=0,              
    tree_method= 'gpu_hist',
    predictor= 'gpu_predictor',)
#3. 훈련
start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_test, y_test)],
          eval_metric='mlogloss',          
          early_stopping_rounds=20
          )
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