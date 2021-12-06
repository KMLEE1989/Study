from enum import auto
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.datasets import load_boston 
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
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
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=66)

model = Sequential() 
model.add(Dense(40, input_dim=13))
model.add(Dense(30))
model.add(Dense(20))
model.add(Dense(10))
model.add(Dense(1))
# model.summary()

#model.load_weights("./_save/kreas25_3_save_weights.h5")

#model.save("./_save/keras25_1_save_model.h5")
#model.save_weights("./_save/keras25_1_save_weights.h5")


model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
es= EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, 
                  restore_best_weights=True
                  )
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath='./_ModelCheckPoint/keras26_4_MCP.hdf5')


start=time.time()
hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp])  #로스와 발로스 반환
end=time.time() - start

print("=======================================================================================")
print(hist)
print("=======================================================================================")
print(hist.history)
print("=======================================================================================")
print(hist.history['loss'])
print("=======================================================================================")
print(hist.history['val_loss'])
print("=======================================================================================")

model.save("./study/_ModelCheckPoint/keras26_4_MCP.hdf5")

import matplotlib.pyplot as plt 

#plt.scatter(x, y)
plt.figure(figsize=(9,6))
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

print("걸린시간 :" , round(end, 3), '초')

#model.save("../study/_save/keras26_1_save_model.h5")
#model.save_weights("./_save/keras25_3_save_weights.h5")
#model.load_weights("./_save/keras25_3_save_weights.h5")


#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

# loss : 36.89948654174805
# r2스코어: 0.5533674412160381
