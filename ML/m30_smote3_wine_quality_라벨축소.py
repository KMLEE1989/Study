# 그냥 증폭해서 성능비교
# from tkinter.tix import LabelEntry
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
from imblearn.over_sampling import SMOTE

path = "../_data/yoondata/"    

datasets = pd.read_csv(path + 'winequality-white.csv', sep=';',index_col=None, header=0)

datasets = datasets.values

x= datasets[:, :11]
y= datasets[:, 11]
# print(y.shape) #(4898,)

print("라벨: ", np.unique(y, return_counts=True))

for index, value in enumerate(y):
    if value == 9 :
        y[index] = 9
    elif value == 8 :
        y[index] = 8
    elif value == 7 :
        y[index] = 7
    elif value == 6 :
        y[index] = 6
    elif value == 5 :
        y[index] = 5
    elif value == 4 :
        y[index] = 4
    elif value == 3 :
        y[index] = 3
    else:
        y[index] = 0
    
         
print(pd.Series(y).value_counts())    

# newlist = []
# for i in y_new:
#     #print(i)
#     if i<=4 :
#         newlist += [0]    
#     elif i<=7:
#         newlist += [1]
#     else:
#         newlist += [2]
            
# y_new=np.array(newlist)  

# print(np.unique(y, return_counts=True))       
        
# x=datasets.drop("quality", axis=1)
# y=datasets["quality"]
# print(x.shape)

# y=np.where(y == 9,8,y)
# y=np.where(y == 7,8,y)
# y=np.where(y == 6,2,y)
# y=np.where(y == 5,2,y)
# y=np.where(y == 4,1,y)
# y=np.where(y == 3,1,y)
# y=np.where(y == 2,1,y)

# print(np.unique((y)))
# print(np.unique(y, return_counts=True))

#print(datasets.info())
#type(datasets)

# for col1 in datasets.columns:
#     n_nan1 = datasets[col1].isnull().sum()
#     if n_nan1>0:
#       msg1 = '{:^20}에서 결측치 개수: {}개'.format(col1,n_nan1)
#       print(msg1)
#     else:
#         print('결측치가 없습니다.')

# for col2 in datasets.columns:
#     n_nan2 = datasets[col2].isnull().sum()
#     if n_nan2>0:
#         msg2 = '{:^20}에서 결측치 개수 : {}개'.format(col2,n_nan2)
#         print(msg2)
#     else:
#       print('결측치가 없습니다.')

# count_data=datasets.groupby('quality')['quality'].count()
# print(count_data)
# plt.bar(count_data.index, count_data)
# plt.show()

# g1 = datasets.groupby( [ "quality"] ).count()
# g1.plot(kind='bar', rot=0)
# plt.show()

# datasets.groupby( [ "quality"] ).count().plot(kind='bar', rot=0)
# plt.show()

# print(datasets.head())
# print(datasets.shape) #(4898, 12)
# print(datasets.describe())
# print(datasets.info())
###############################그래프 그려봐!!!##########################################################
#pd.value_counts
#group by 쓰구, count() 써라.

#plt.bar 로 그려라 quality 


#########################################################################################################

#       fixed acidity  volatile acidity  citric acid  residual sugar    chlorides  free sulfur dioxide  total sulfur dioxide      density           pH    sulphates      alcohol      quality
# count    4898.000000       4898.000000  4898.000000     4898.000000  4898.000000          4898.000000           4898.000000  4898.000000  4898.000000  4898.000000  4898.000000  4898.000000      
# mean        6.854788          0.278241     0.334192        6.391415     0.045772            35.308085            138.360657     0.994027     3.188267     0.489847    10.514267     5.877909      
# std         0.843868          0.100795     0.121020        5.072058     0.021848            17.007137             42.498065     0.002991     0.151001     0.114126     1.230621     0.885639      
# min         3.800000          0.080000     0.000000        0.600000     0.009000             2.000000              9.000000     0.987110     2.720000     0.220000     8.000000     3.000000      
# 25%         6.300000          0.210000     0.270000        1.700000     0.036000            23.000000            108.000000     0.991723     3.090000     0.410000     9.500000     5.000000      
# 50%         6.800000          0.260000     0.320000        5.200000     0.043000            34.000000            134.000000     0.993740     3.180000     0.470000    10.400000     6.000000      
# 75%         7.300000          0.320000     0.390000        9.900000     0.050000            46.000000            167.000000     0.996100     3.280000     0.550000    11.400000     6.000000      
# max        14.200000          1.100000     1.660000       65.800000     0.346000           289.000000            440.000000     1.038980     3.820000     1.080000    14.200000     9.000000      

# datasets=datasets.values #넘파이로 바꾸잡!
# print(type(datasets))
# print(datasets.shape)


# x = datasets[:, :11]
# y = datasets[:, 11]

#print("라벨: ", np.unique(y, return_counts=True))
# 라벨:  (array([3., 4., 5., 6., 7., 8., 9.]), array([  20,  163, 1457, 2198,  880,  175,    5], dtype=int64))

# #################### 아웃라이어확인 #################

# def boxplot_vis(data, target_name):
#     plt.figure(figsize=(30, 30))
#     for col_idx in range(len(data.columns)):
#         # 6행 2열 서브플롯에 각 feature 박스플롯 시각화
#         plt.subplot(6, 2, col_idx+1)
#         # flierprops: 빨간색 다이아몬드 모양으로 아웃라이어 시각화
#         plt.boxplot(data[data.columns[col_idx]], flierprops = dict(markerfacecolor = 'r', marker = 'D'))
#         # 그래프 타이틀: feature name
#         plt.title("Feature" + "(" + target_name + "):" + data.columns[col_idx], fontsize = 20)
#     # plt.savefig('../figure/boxplot_' + target_name + '.png')
#     plt.show()
# boxplot_vis(datasets,'white_wine')

# #################### 아웃라이어 처리 ###############################

# def remove_outlier(input_data):
#     q1 = input_data.quantile(0.25) # 제 1사분위수
#     q3 = input_data.quantile(0.75) # 제 3사분위수
#     iqr = q3 - q1 # IQR(Interquartile range) 계산
#     minimum = q1 - (iqr * 1.5) # IQR 최솟값
#     maximum = q3 + (iqr * 1.5) # IQR 최댓값
#     # IQR 범위 내에 있는 데이터만 산출(IQR 범위 밖의 데이터는 이상치)
#     df_removed_outlier = input_data[(minimum < input_data) & (input_data < maximum)]
#     return df_removed_outlier

# prep = remove_outlier(datasets)

# prep['target'] = 0

# a = prep.isnull().sum()
# print(a)

# prep.dropna(axis = 0, how='any', inplace = True)
# print(f"이상치 포함된 데이터 비율: {round((len(datasets) - len(prep))*100/len(datasets), 2)}%")

# x=prep.drop(['quality'], axis=1)
# y=prep['quality']

# print(x.shape, y.shape)

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

x_train, x_test,y_train,y_test = train_test_split(
    x,y,random_state=66, shuffle=True, train_size=0.8, stratify=y)

smote=SMOTE(random_state=66, k_neighbors=2)

x_train, y_train = smote.fit_resample(x_train, y_train)

scaler = MinMaxScaler()
x_train=scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

model=XGBClassifier(n_estimators=1000,
    learning_rate=0.1,               
    max_depth=8,
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

y_predict = model.predict(x_test)

score = model.score(x_test, y_test)
print('model.score : ', round(score, 4))
print('acc_score : ', round(accuracy_score(y_test, y_predict),4))
print('f1_score : ', round(f1_score(y_test, y_predict, average='macro'), 4))
# print('f1_score : ', f1_score(y_test, y_predict, average='micro'))
# print('f1_score : ', f1_score(y_test, y_predict, average='weighted'))

# 걸린시간 :  6.395215749740601
# model.score :  0.613265306122449
# acc_score :  0.613265306122449
# f1_score :  0.37193689485014947
# f1_score :  0.613265306122449
# f1_score :  0.6137298438612334


# smote 전
# 걸린시간 :  4.117445945739746
# model.score :  0.6173
# acc_score :  0.6173
# f1_score :  0.3274

# smote 후
# 걸린시간 :  6.3792946338653564
# model.score :  0.6204
# acc_score :  0.6204
# f1_score :  0.3903

# print("====================================SMOTE 적용=============================")

# smote=SMOTE(random_state=66, k_neighbors=3)

# x_train,y_train = smote.fit_resample(x_train, y_train)

# print(pd.Series(y_train).value_counts())
# # 1    53
# # 0    44
# # 2    14
# model=XGBClassifier(n_jobs=4)

# model.fit(x_train, y_train)

# score=model.score(x_test, y_test)

# y_predict = model.predict(x_test)

# print("model.score: ", round(score, 4))
# print('acc_score : ', accuracy_score(y_test, y_predict))
# print('f1_score : ', f1_score(y_test, y_predict, average='macro'))
# print('f1_score : ', f1_score(y_test, y_predict, average='micro'))
# print('f1_score : ', f1_score(y_test, y_predict, average='weighted'))

# y_predict = model.predict(x_test)

# print("accuracy score : ", 
#       round(accuracy_score(y_test, y_predict),4))

# model.score:  0.6418
# acc_score :  0.6418367346938776
# f1_score :  0.39203962714739304
# f1_score :  0.6418367346938776
# f1_score :  0.6388455583957404
# accuracy score :  0.6418

# 라벨축소 smote
# 걸린시간 :  4.092931270599365
# model.score :  0.9224
# acc_score :  0.9224
# f1_score :  0.6399

