<<<<<<< HEAD
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_train=np.array([1,2,3,4,5,6,7])
y_test=np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #1 eopchs당 7번 돈다   100*7전체훈련 배치가 2이면 400 (1,2:1,2  3,4:3,4 5,6:5,6 7:7)*100

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict([11])
print('11의 예측값: ', result)
=======
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np  

#1. 데이터
x_train=np.array([1,2,3,4,5,6,7])
x_test=np.array([8,9,10])
y_train=np.array([1,2,3,4,5,6,7])
y_test=np.array([8,9,10])


#2. 모델구성
model = Sequential()
model.add(Dense(10, input_dim=1))
model.add(Dense(1))

#3. 컴파일
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1) #1 eopchs당 7번 돈다   100*7전체훈련 배치가 2이면 400 (1,2:1,2  3,4:3,4 5,6:5,6 7:7)*100

#4. 평가,예측
loss = model.evaluate(x_test, y_test)
print('loss:', loss)
result = model.predict([11])
print('11의 예측값: ', result)
>>>>>>> 8f30fb32eeacf490723b49c3cf100847fbe1fe0d
