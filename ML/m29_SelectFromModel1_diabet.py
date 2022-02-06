import numpy as np 
import pandas as pd 

from xgboost import XGBRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectFromModel

#1. 데이터
# x, y = load_diabetes(return_X_y=True)
# print(x.shape, y.shape) #(442, 10) (442,)

datasets=load_diabetes()
x=datasets.data
y=datasets.target

print(datasets.feature_names)
['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']

x = np.delete(x, [0,1,4,7], axis=1)

x_train, x_test,y_train,y_test = train_test_split(
    x,y,random_state=66, shuffle=True, train_size=0.8)

scaler = StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test =scaler.transform(x_test)

model=XGBRegressor(n_jobs = -1,
                     n_estimators = 2700, 
                     learning_rate =0.35, 
                     max_depth = 7,         
                     min_child_weight = 1,  
                     subsample = 1,         
                     colsample_bytree = 1, 
                     reg_alpha = 1,      
                     reg_lambda = 0,)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print('model.score : ', score)

print(model.feature_importances_)
print(np.sort(model.feature_importances_))
threshold = np.sort(model.feature_importances_)
# [0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
#  0.06597802 0.07382318 0.19681741 0.39979857]

print("==============================================================")

for thresh in threshold:
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    print(select_x_train.shape, select_x_test.shape)

    selection_model = XGBRegressor(n_jobs=-1)
    selection_model.fit(select_x_train, y_train)
    
    y_predict = selection_model.predict(select_x_test)
    score = r2_score(y_test, y_predict)
    
    print("Thresh=%.3f, n=%d, R2: %.2f%%"
          %(thresh, select_x_train.shape[1], score*100))

#피처 줄인다음에!!!
#다시 모델해서 결과 비교

'''
model.score :  0.2396187053539609
[0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739
 0.07382318 0.03284872 0.39979857 0.06597802]
[0.02593721 0.03284872 0.03821949 0.04788679 0.05547739 0.06321319
 0.06597802 0.07382318 0.19681741 0.39979857]
==============================================================
(353, 10) (89, 10)
Thresh=0.026, n=10, R2: 23.96%
(353, 9) (89, 9)
Thresh=0.033, n=9, R2: 27.03%
(353, 8) (89, 8)
Thresh=0.038, n=8, R2: 23.87%
(353, 7) (89, 7)
Thresh=0.048, n=7, R2: 26.48%
(353, 6) (89, 6)
Thresh=0.055, n=6, R2: 30.09%
(353, 5) (89, 5)
Thresh=0.063, n=5, R2: 27.41%
(353, 4) (89, 4)
Thresh=0.066, n=4, R2: 29.84%
(353, 3) (89, 3)
Thresh=0.074, n=3, R2: 23.88%
(353, 2) (89, 2)
Thresh=0.197, n=2, R2: 14.30%
(353, 1) (89, 1)
Thresh=0.400, n=1, R2: 2.56%

'''
'''
(353, 6) (89, 6)
Thresh=0.088, n=6, R2: 30.09%
(353, 5) (89, 5)
Thresh=0.091, n=5, R2: 21.68%
(353, 4) (89, 4)
Thresh=0.095, n=4, R2: 26.79%
(353, 3) (89, 3)
Thresh=0.097, n=3, R2: 18.47%
(353, 2) (89, 2)
Thresh=0.219, n=2, R2: 14.30%
(353, 1) (89, 1)
Thresh=0.410, n=1, R2: 2.56%
'''