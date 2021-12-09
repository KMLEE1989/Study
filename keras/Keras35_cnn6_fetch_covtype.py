from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

datasets = fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data           
y = datasets.target 

print(x.shape) #(581012, 54)
print(y.shape) #(581012,)

print(np.unique(y)) #[1 2 3 4 5 6 7]

x_train,x_test,y_train,y_test = train_test_split(x,y, train_size=0.8, shuffle=True, random_state=49)

print(x_train.shape, x_test.shape) #(464809, 54) (464809, 54)

scaler = MinMaxScaler() 
x_train = scaler.fit_transform(x_train).reshape(464809, 6,9,1)
x_test = scaler.transform(x_test).reshape(116203, 6,9,1)

import pandas as pd 
y_train=pd.get_dummies(y_train)
y_test=pd.get_dummies(y_test)

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(6,9,1), activation='relu'))                                                                              # 1,1,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       
model.add(MaxPooling2D(2,2))                                                                                    
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(7))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)

model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

# loss :  0.056066203862428665
# r2스코어 :  0.2545237336560864