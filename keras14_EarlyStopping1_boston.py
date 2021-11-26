from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
import time

datasets = load_boston()
x=datasets.data
y=datasets.target

'''
print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)
'''
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=49)

model = Sequential() 
model.add(Dense(50, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(8))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)


start=time.time()
hist = model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.5, callbacks=[es])  #로스와 발로스 반환
end=time.time() - start

print("걸린시간:  ", round(end,3), '초')

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

'''
print("=======================================================================================")
print(hist)
print("=======================================================================================")
print(hist.history)
print("=======================================================================================")
print(hist.history['loss'])
print("=======================================================================================")
print(hist.history['val_loss'])
print("=======================================================================================")
'''

import matplotlib.pyplot as plt 

#plt.scatter(x, y)
plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

# Epoch 00291: early stopping
# 걸린시간:   48.859 초
# 5/5 [==============================] - 0s 0s/step - loss: 39.3702
# loss : 39.37022018432617
# r2스코어: 0.49045458362964023
