#import tensorflow as tf
from tensorflow.keras.models import Sequential  
from tensorflow.python.keras.layers import Dense 
import numpy as np

#1.데이터 
x = np.array([1,2,3])  
y = np.array([1,2,3])

#2. 모델 구성
model = Sequential()   
model.add(Dense(5, input_dim=1)) 
model.add(Dense(3, activation='relu'))  
model.add(Dense(4, activation='sigmoid'))  
model.add(Dense(2, activation='relu'))  
model.add(Dense(1))

model.summary()

'''
#1+1#(바이어스)

#5+1#(바이어스)  2X5(param)=10

#3+1(바이어스)   6X3(param)=18

#4+1(바이어스)   4X4(param)=16

#2+1(바이어스)   5X2(param)=10

#1+1(바이어스)   3X1(param)=3

#################################################### 바이어스의 존재를 기억할것!


_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
dense (Dense)                (None, 5)                 10
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 18
_________________________________________________________________
dense_2 (Dense)              (None, 4)                 16
_________________________________________________________________
dense_3 (Dense)              (None, 2)                 10
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 3
=================================================================
Total params: 57  
Trainable params: 57
Non-trainable params: 0 (훈련되지 않은 가중치의 갯수)


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')  

model.fit(x, y, epochs=30, batch_size=1) 


#4. 평가,예측
loss = model.evaluate(x,y)  
print('loss: ',loss)
result=model.predict([4]) 
print('4의 예측값: ', result)

'''


