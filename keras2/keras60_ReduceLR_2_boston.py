from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

#1. 데이터
datasets = load_boston()
x = datasets.data
y = datasets.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.8, random_state=42)

scaler =  MinMaxScaler()         
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델구성
model = Sequential()
model.add(Dense(100, input_dim=13)) 
model.add(Dense(80, activation='relu'))
model.add(Dense(130, activation='sigmoid'))
model.add(Dense(80, activation='relu'))
model.add(Dense(5))
model.add(Dense(1))

#3. 컴파일, 훈현
from tensorflow.keras.optimizers import Adam, Adadelta, Adagrad, Adamax
from tensorflow.keras.optimizers import RMSprop, SGD, Nadam

learning_rate = 0.001
optimizer = Adam(lr=learning_rate)
model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])   # loss에 들어가는 지표는 성능에 영향을 미치지만, metrics에 들어가는 지표는 성능에 영향을 미치지 않음(단순평가지표)

import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=15, mode='auto', verbose=1, factor=0.5) 
 
start = time.time()
model.fit(x_train, y_train, epochs=500, batch_size=13, callbacks=[es, reduce_lr])
end = time.time() - start

#4. 평가, 예측
loss, mae = model.evaluate(x_test, y_test)
print('learning_rate : ', learning_rate)
print('loss : ', round(loss,4))
print('mae : ', round(mae,4))
print('걸린시간 :', round(end,4))

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2스코어 : ', r2)

# learning_rate :  0.001
# loss :  18.8911
# mae :  3.3795
# 걸린시간 : 82.0666
# r2스코어 :  0.7713411930351318

'''
loss : 19.184654235839844
r2스코어 :  0.7677883338121207
'''

'''
loss : 17.076885223388672
r2스코어 :  0.7933008109604748
'''

'''
loss : 19.80939292907715
r2스코어 :  0.7602264967945693
'''