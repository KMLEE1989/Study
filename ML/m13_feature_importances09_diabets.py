
import numpy as np 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_diabetes()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

model1 = DecisionTreeRegressor(max_depth=5)
model2= RandomForestRegressor(max_depth=5)
model3 = XGBRegressor()
model4 = GradientBoostingRegressor()

#3) 훈련

model1.fit(x_train, y_train) 
model2.fit(x_train, y_train) 
model3.fit(x_train, y_train) 
model4.fit(x_train, y_train) 

#4) 평가, 예측
result1 = model1.score(x_test, y_test) 
result2 = model2.score(x_test, y_test) 
result3 = model3.score(x_test, y_test) 
result4 = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score, r2_score
y_predict1 = model1.predict(x_test)
r2_1 = r2_score(y_test, y_predict1) 
y_predict2 = model2.predict(x_test)
r2_2 = r2_score(y_test, y_predict2) 
y_predict3 = model3.predict(x_test)
r2_3 = r2_score(y_test, y_predict3) 
y_predict4 = model4.predict(x_test)
r2_4 = r2_score(y_test, y_predict4) 

print("accuracy: ", r2_1)  #accuracy:  0.18699053453135217
print("accuracy: ", r2_2)  #accuracy:  0.4072630516607163
print("accuracy: ", r2_3)  #accuracy:  0.23802704693460175
print("accuracy: ", r2_4)  #accuracy:  0.38874082366191653

print(model1.feature_importances_) 
# [0.04339214 0.         0.24919201 0.11505227 0.         0.04366568
#  0.0403601  0.         0.45459058 0.05374722] 
print(model2.feature_importances_) 
# [0.044875   0.00729148 0.32083246 0.10594678 0.02322774 0.03629835
#  0.03210046 0.01544369 0.35196423 0.0620198 ]
print(model3.feature_importances_) 
# [0.02593721 0.03821949 0.19681741 0.06321319 0.04788679 0.05547739
#  0.07382318 0.03284872 0.39979857 0.06597802]
print(model4.feature_importances_)  
# [0.0600784  0.01147184 0.2760129  0.11755902 0.02311864 0.05539218
#  0.03913221 0.01630758 0.34267998 0.05824725]
import matplotlib.pyplot as plt

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
             align = 'center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Feautures")
    plt.ylim(-1, n_features)
    
plt.subplot(2,2,1)  # 2행 2열로 뽑아내라 첫번째꺼를
plot_feature_importances_dataset(model1)
plt.subplot(2,2,2)
plot_feature_importances_dataset(model2)
plt.subplot(2,2,3)
plot_feature_importances_dataset(model3)
plt.subplot(2,2,4)
plot_feature_importances_dataset(model4)
plt.show()