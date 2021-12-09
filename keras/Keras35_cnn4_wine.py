from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data           
y = datasets.target 

print(x.shape) #(178, 13)
print(y.shape) #(178,)

print(np.unique(y))

xx = pd.DataFrame(x, columns=datasets.feature_names)

xx['class'] = y 

import matplotlib.pyplot as plt
import seaborn as sns   # 조금 더 이쁘게 만들게 도와줌.
plt.figure(figsize=(10,10))
sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# seaborn heatmap 개념정리
plt.show()


xx= xx.drop(['flavanoids','class'], axis=1) 

xx = xx.values 

x_train,x_test,y_train,y_test = train_test_split(xx,y, train_size=0.8, shuffle=True, random_state=49)

scaler =MinMaxScaler() 
x_train = scaler.fit_transform(x_train).reshape(len(x_train),2,2,3)
x_test = scaler.transform(x_test).reshape(len(x_test),2,2,3)


model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(2,2,3), activation='relu'))    # 2,2,10                                                                           # 1,1,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 2,2,10 
model.add(MaxPooling2D(2,2))                                                                                # 1,1,10     
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)
model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])


loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)

# loss :  0.04398845136165619
# r2스코어 :  0.9208207853602509


