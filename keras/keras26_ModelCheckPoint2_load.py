from enum import auto
from tensorflow.keras.models import Sequential, load_model
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

# model = Sequential() 
# model.add(Dense(40, input_dim=13))
# model.add(Dense(30))
# model.add(Dense(20))
# model.add(Dense(10))
# model.add(Dense(1))


# model.compile(loss='mse', optimizer='adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint  
# es= EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1, restore_best_weights=True)
# mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
#                     save_best_only=True,
#                     filepath='./_ModelCheckPoint/keras26_1_MCP.hdf5')


# start=time.time()
# hist = model.fit(x_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[es,mcp])  #로스와 발로스 반환
# end=time.time() - start


# print("=======================================================================================")
# print(hist.history['val_loss'])
# print("=======================================================================================")


# print("걸린시간 :" , round(end, 3), '초')

# model.save("../study/_save/keras26_1_save_model.h5")

#model=load_model('../study/_save/keras26_1_save_model.h5')
model=load_model("./study/_ModelCheckPoint/keras26_4_MCP.hdf5")

#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score   #save weights 할때 컴파일 할때 해야한다.
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)


# loss : 52.6992301940918
# r2스코어: 0.3621267923488489

# loss : 60.422767639160156
# r2스코어: 0.26864101023692855