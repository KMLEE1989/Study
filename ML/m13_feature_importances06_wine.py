import numpy as np 
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_wine()

x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state=66)

#print(x_train.shape, y_train.shape)  #(120, 4) (120, 3)
#print(x_test.shape, y_test.shape)    #(30, 4) (30, 3)

#2) 모델구성 
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

model1 = DecisionTreeClassifier(max_depth=5)
model2= RandomForestClassifier(max_depth=5)
model3 = XGBClassifier()
model4 = GradientBoostingClassifier()

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

from sklearn.metrics import accuracy_score
y_predict1 = model1.predict(x_test)
acc1 = accuracy_score(y_test, y_predict1) 
y_predict2 = model2.predict(x_test)
acc2 = accuracy_score(y_test, y_predict2) 
y_predict3 = model3.predict(x_test)
acc3 = accuracy_score(y_test, y_predict3) 
y_predict4 = model4.predict(x_test)
acc4 = accuracy_score(y_test, y_predict4) 

print("accuracy: ", acc1)  #accuracy:  0.9722222222222222
print("accuracy: ", acc2)  #accuracy:  1.0 
print("accuracy: ", acc3)  #accuracy:  1.0
print("accuracy: ", acc4)  #accuracy:  0.9722222222222222

print(model1.feature_importances_)  
# [0.00489447 0.         0.         0.         0.01598859 0.
#  0.1569445  0.         0.         0.04078249 0.08604186 0.33215293
#  0.36319516]
print(model2.feature_importances_) 
# [0.12337028 0.0324303  0.0165727  0.0387889  0.02743616 0.06228313
#  0.1190847  0.00955604 0.02027061 0.13621221 0.09725919 0.14799275
#  0.16874304]
print(model3.feature_importances_) 
# [0.01854127 0.04139537 0.01352911 0.01686821 0.02422602 0.00758254
#  0.10707159 0.01631111 0.00051476 0.12775213 0.01918284 0.50344414
#  0.10358089]
print(model4.feature_importances_)  
# [1.74539485e-02 3.94054140e-02 1.99910896e-02 6.83983587e-03
#  9.33266965e-04 3.18802210e-05 1.06765085e-01 6.61073412e-04
#  1.21112468e-04 2.46898447e-01 3.36221997e-02 2.48734759e-01
#  2.78541888e-01]
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
