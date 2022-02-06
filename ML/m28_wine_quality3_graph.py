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

path = "../_data/yoondata/"    

datasets = pd.read_csv(path + 'winequality-white.csv', sep=';',index_col=None, header=0)

#print(datasets.info())

#type(datasets)
print(datasets.head())
print(datasets.describe())   # pandas에서 볼수있는기능 (수치데이터에서 용이함)
print(datasets.info())
# print(data.columns)
# print(type(data))   # <class 'pandas.core.frame.DataFrame'>

x = datasets.drop(['quality'], axis=1)  
# print(x.shape)  # (4898, 11)
y = datasets['quality']
# print(y.shape)  # (4898,)

# print(np.unique(y, return_counts=True))  # (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

####################################### 그래프 그리기 ######################################
# pd.value_counts -> 이거 쓰지말기
# groupby와 count()를 이용하여 그리기
# plt.bar로 그리기 quality

import matplotlib.pyplot as plt

# g1 = data.groupby( [ "quality"] ).count()
# g1.plot(kind='bar', rot=0)
# plt.show()

# data.groupby( [ "quality"] ).count().plot(kind='bar', rot=0)
# plt.show()

count_data = datasets.groupby('quality')['quality'].count()
print(count_data)
plt.bar(count_data.index, count_data)
plt.show()
#################################################################################################

# def outliers(data_out):
#     quartile_1, q2, quartile_3 = np.percentile(data_out, [25,50,75])
#     print("1사분위 : ", quartile_1)
#     print("q2 : ", q2)
#     print("3사분위 : ", quartile_3)
#     iqr = quartile_3 - quartile_1
#     print("iqr : ", iqr)
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     return np.where((data_out>upper_bound) | (data_out<lower_bound))    

# outliers_loc = outliers(data)
# print("이상치의 위치 : ", outliers_loc)

# import matplotlib.pyplot as plt
# plt.boxplot(data)
# plt.show()

# x_train, x_test, y_train, y_test = train_test_split(x, y,
#                                                     train_size=0.8, shuffle=True, random_state=66, stratify=y)

# scaler = StandardScaler()
# x_train = scaler.fit_transform(x_train)
# x_test = scaler.transform(x_test)

# #2. 모델
# model = XGBClassifier(
#                      n_estimators=10000,   # n_estimators는 epoch와 같은 의미
#                      n_jobs=-1,           
#                      learning_rate=0.005, 
#                      max_depth=13,         
#                      min_child_weight=1,
#                      subsample=1,
#                      colsample_bytree=1,
#                      reg_alpha=0,         # 규제  L1  
#                      reg_lambda=1         # 규제  L2
#                      )     

# #3. 훈련
# start = time.time()
# model.fit(x_train, y_train, verbose=1,
#           eval_set = [(x_test, y_test)],
#           eval_metric= 'merror',        #rmse, mae, logloss, error
#           early_stopping_rounds=900 
#           )  # mlogloss
# end = time.time()

# results = model.score(x_test, y_test)
# print("results: ", round(results, 4))

# y_predict = model.predict(x_test)
# acc = accuracy_score(y_test, y_predict)
# print("accuracy_score : ", round(acc, 4))
# print("f1_score : ", f1_score(y_test, y_predict, average='macro'))
# print("f1_score : ", f1_score(y_test, y_predict, average='micro'))

# print("걸린시간: ", end - start)


# print("=====================================")

'''
results:  0.7102
accuracy_score :  0.7102
걸린시간:  38.32083296775818
results:  0.6745
accuracy_score :  0.6745
f1_score :  0.4523239916472014
걸린시간:  24.807340145111084
'''