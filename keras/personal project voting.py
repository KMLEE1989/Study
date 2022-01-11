import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from pandas import set_option 
from pandas.plotting import scatter_matrix 
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV 
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, ElasticNet 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR 
from sklearn.pipeline import Pipeline 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor 
from sklearn.metrics import mean_squared_error 
from xgboost import XGBRegressor, XGBClassifier 
from lightgbm import LGBMRegressor, LGBMClassifier 
from catboost import CatBoostRegressor, CatBoostClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn import preprocessing 
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, MaxAbsScaler 
import seaborn as sns 
 
#1. 데이터 
path = "../_data/개인프로젝트/CSV/"
dataset = pd.read_csv(path + "통합.csv") 

#print(dataset.info()) 


data = dataset.drop(['DATE', '지점','NUM','Domestic','MEN', 'WOMEN','INSPECTION'], axis=1)
target = dataset['AVG TEMP(℃)']

x = dataset.drop(['DATE', '지점','NUM','Domestic','MEN', 'WOMEN','INSPECTION'],axis=1)
y = dataset['AVG TEMP(℃)'] 

print(x.info()) 

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66) 

model1 = RandomForestRegressor(n_estimators = 100, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model2 = GradientBoostingRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model3 = ExtraTreesRegressor(n_estimators=100, max_depth=16, random_state=7) 
model4 = AdaBoostRegressor(n_estimators=100, random_state=7) 
model5 = XGBRegressor(n_estimators = 100, learning_rate = 0.3, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model6 = LGBMRegressor(n_estimators = 100, learning_rate = 0.1, max_depth=10, min_samples_split=40, min_samples_leaf =30) 
model7 = CatBoostRegressor(n_estimators=100, max_depth=16, random_state=7) 
 
from sklearn.ensemble import VotingClassifier 
voting_model = VotingClassifier(estimators=[('RandomForestRegressor', model1), 
                                            ('GradientBoostingRegressor', model2), 
                                            ('ExtraTreesRegressor', model3), 
                                            ('AdaBoostRegressor', model4), 
                                            ('XGBRegressor', model5), 
                                            ('LGBMRegressor', model6), 
                                            ('CatBoostRegressor', model7)], voting='hard') 
 
classifiers = [model1,model2,model3,model4,model5,model6,model7] 
from sklearn.metrics import r2_score 
 
for classifier in classifiers: 
    classifier.fit(x_train, y_train) 
    y_predict = classifier.predict(x_test) 
    r2 = r2_score(y_test, y_predict) 
           
    class_name = classifier.__class__.__name__ 
    print("============== " + class_name + " ==================") 
     
    print('r2 스코어 : ', round(r2,3)) 
    print('예측값 : ', y_predict[-1]) 

# ============== RandomForestRegressor ==================
# r2 스코어 :  0.977
# 예측값 :  23.11932308872073
# ============== GradientBoostingRegressor ==================
# r2 스코어 :  0.992
# 예측값 :  24.699366470980436
# ============== ExtraTreesRegressor ==================
# r2 스코어 :  1.0
# 예측값 :  24.423999999999996
# ============== AdaBoostRegressor ==================
# r2 스코어 :  0.997
# 예측값 :  25.071641791044772

# ============== XGBRegressor ==================
# r2 스코어 :  1.0
# 예측값 :  24.41864
# [LightGBM] [Warning] Unknown parameter: min_samples_split
# [LightGBM] [Warning] min_data_in_leaf is set with min_child_samples=20, will be overridden by min_samples_leaf=30. Current value: min_data_in_leaf=30
# ============== LGBMRegressor ==================
# r2 스코어 :  0.994
# 예측값 :  24.421388374438276

