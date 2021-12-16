import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Conv1D,Flatten
import time

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape)  #(178, 13) (178,)
print(np.unique(y))  #[0 1 2]

y=to_categorical(y)
print(y)
print(y.shape)  #(178, 3)

x=x.reshape(178,13,1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential() 
model.add(Conv1D (50, 2, input_shape=(13,1))) 
model.add(Flatten())
model.add(Dense(55))
model.add(Dense(60, activation='relu'))
model.add(Dense(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:5])
print(y_test[:5])
print(results)


# Conv1D
# 걸린시간:   3.58
# loss : 0.1534937471151352
# accuracy: 0.9444444179534912