import numpy as np 
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
#1) 데이터
datasets = load_breast_cancer()

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

print("accuracy: ", acc1)  #accuracy:   0.9298245614035088
print("accuracy: ", acc2)  #accuracy:   0.956140350877193 
print("accuracy: ", acc3)  #accuracy:  0.9736842105263158
print("accuracy: ", acc4)  #accuracy:   0.9473684210526315

print(model1.feature_importances_)  
# [0.         0.06054151 0.         0.         0.         0.02291518
#  0.         0.02005078 0.         0.         0.00716099 0.00636533
#  0.         0.01257413 0.         0.         0.004774   0.00442037
#  0.         0.         0.         0.01642816 0.         0.72839202
#  0.         0.         0.         0.11637753 0.         0.        ]
print(model2.feature_importances_) 
# [0.04754348 0.01700255 0.05339459 0.04024622 0.00586093 0.01108472
#  0.04161256 0.11014981 0.00298738 0.00384568 0.01353678 0.00429022
#  0.01090013 0.06241891 0.00245134 0.00314242 0.0024851  0.00553003
#  0.00386488 0.00374494 0.10505515 0.01276166 0.14593152 0.09915046
#  0.01179347 0.01343393 0.03594601 0.11348893 0.0095277  0.00681851]
print(model3.feature_importances_) 
# [0.01420499 0.03333857 0.         0.02365488 0.00513449 0.06629944
#  0.0054994  0.09745206 0.00340272 0.00369179 0.00769183 0.00281184
#  0.01171023 0.0136856  0.00430626 0.0058475  0.00037145 0.00326043
#  0.00639412 0.0050556  0.01813928 0.02285904 0.22248559 0.2849308
#  0.00233393 0.         0.00903706 0.11586287 0.00278498 0.00775311]
print(model4.feature_importances_) 
# [2.47621446e-05 3.75160469e-02 4.53971223e-04 2.00718949e-03
#  9.49556494e-04 4.25116905e-03 1.97006602e-03 1.30141235e-01
#  3.57862377e-03 6.68689240e-04 3.98009660e-03 9.87128983e-06
#  9.08439002e-04 1.78917658e-02 1.42684754e-03 2.53269919e-03
#  9.20614996e-04 7.43672794e-04 3.16479355e-05 2.60322995e-03
#  3.11267426e-01 4.16465424e-02 3.64344773e-02 2.74150059e-01
#  3.34733036e-03 1.29887239e-04 1.35740798e-02 1.06550338e-01
#  2.27256244e-05 2.66939600e-04] 

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


