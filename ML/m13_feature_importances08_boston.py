import numpy as np 
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_boston()

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

print("accuracy: ", r2_1)  #accuracy:  0.8507309980875365
print("accuracy: ", r2_2)  #accuracy:  0.9112497201379339
print("accuracy: ", r2_3)  #accuracy:  0.9221188601856797
print("accuracy: ", r2_4)  #accuracy:  0.9457670522088083

print(model1.feature_importances_)  
# [0.02912853 0.         0.         0.         0.01336691 0.29092518
#  0.         0.06904396 0.         0.02369397 0.         0.
#  0.57384145]
print(model2.feature_importances_) 
# [3.57668971e-02 2.31780519e-04 4.12964874e-03 1.09768039e-03
#  1.83556388e-02 4.46327887e-01 5.58687816e-03 5.82892687e-02
#  2.42842787e-03 1.08196427e-02 1.27725566e-02 5.77853324e-03
#  3.98415160e-01]
print(model3.feature_importances_) 
# [0.01447933 0.00363372 0.01479118 0.00134153 0.06949984 0.30128664
#  0.01220458 0.05182539 0.0175432  0.03041654 0.04246344 0.01203114
#  0.4284835 ]
print(model4.feature_importances_)  
# [2.38783901e-02 1.63441555e-04 5.03756120e-03 1.79671284e-04
#  4.10599027e-02 3.57845883e-01 6.43841560e-03 8.40724401e-02
#  2.33641816e-03 1.09457645e-02 3.11295407e-02 6.58631569e-03
#  4.30326256e-01]
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