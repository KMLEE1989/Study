import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
 

path = "../_data/개인프로젝트/CSV/"

dft=pd.read_csv(path+'통합.csv',thousands=',')

# print(dft.info())

dft_X=dft.drop(['DATE','지점','AVG TEMP(℃)', 'MAX TEMP(℃)','MIN TEMP(℃)'],axis=1)
dft_Y=dft['AVG TEMP(℃)']

#print(dft_X.isnull().sum()) #dft_X-> 통합 XGBOOST CSV

# MAX TEMP(℃)    0
# MIN TEMP(℃)    0
# PM10           0
#  PM2.5         0
# SO2            0
# O3             0
# NO2            0
# CO             0
# NUM            0
# Domestic       0
# Foreign        0
# MEN            0
# WOMEN          0
# INSPECTION     0
# dtype: int64

dft_x_train, dft_x_test, dft_y_train, dft_y_test=train_test_split(dft_X, dft_Y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsampel=0.75, colsample_bytree=1, max_depth=7)

#print(len(dft_x_train), len(dft_x_test))
xgb_model.fit(dft_x_train, dft_y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1,
             missing=None, n_estimators=1000, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=True, subsample=0.75)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)

plt.show()

pred=xgb_model.predict(dft_x_test)
print(pred[-30:])

list1=[ 6.3407645,   3.2552927,   9.489924,    5.1280856,  17.028868,   16.949871, 22.001284,   11.476369,    6.434142,   15.403754,   10.20475,     2.3318226,
 14.79987,    25.001297,    4.559653,   16.671524,    2.4657805,  -5.389366,
 17.568739,   26.583458,    2.4746122,  10.327534,   10.786948,   23.787401,
 -5.5242767,  20.845575,   27.584475,   25.489971,   20.703094,   22.209776]


r_sq=xgb_model.score(dft_x_train, dft_y_train)
print(r_sq)
print(explained_variance_score(pred, dft_y_test))

plt.figure(figsize=(12, 9))
plt.plot('list1', marker='o')
plt.xlabel('DATE from January 1')
plt.ylabel('AVG Temp')
plt.legend(['S.korea AVG TEMP per day in january'], loc='upper right', fontsize=10)
plt.show()


# 0.9999999349134341
# 0.8498834265549486




