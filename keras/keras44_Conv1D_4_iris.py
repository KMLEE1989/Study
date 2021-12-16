import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import time
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, Conv1D,Flatten


datasets = load_iris()

x=datasets.data
y=datasets.target

#print(x.shape, y.shape)  #(150, 4) (150,)

x=x.reshape(150,4,1)

y=to_categorical(y)
print(y)
print(y.shape)  #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)

print(x_train.shape, y_train.shape) #(120, 4, 1) (120, 3)  
print(x_test.shape, y_test.shape)  #(30, 4, 1) (30, 3)


model = Sequential() 
model.add(Conv1D(10, 2, input_shape=(4,1)))  #(30, 4) (30, 3)
model.add(Flatten())
model.add(Dense(10))
model.add(Dense(70, activation='relu'))
model.add(Dense(60))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])  #얼리스타핑을 콜백으로 땡기겠다 리스트
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])

results=model.predict(x_test[:5])
print(y_test[:5])
print(results)

# Conv1D
# 걸린시간:   2.353
# loss : 0.0782230794429779
# accuracy: 0.9666666388511658

# LSTM
# loss : 0.06429038196802139
# accuracy: 0.9666666388511658