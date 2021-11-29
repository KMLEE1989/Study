from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense
import numpy as np
from sklearn.model_selection import train_test_split

# #1. 데이터
x=np.array(range(1,17))
y=np.array(range(1,17))

#train_test_split로 나누시오 
x_train, x_test, y_train, y_test= train_test_split(x,y,
                 train_size=0.8125, shuffle=True, random_state=66)
#13개 3개 

#x_test, x_val, y_test, y_val= train_test_split(x_test,y_test,
#                 train_size=0.5, shuffle=False, random_state=66)

#x_test, x_val, y_test, y_val=train_test_split(x_test, y_test, test_size=0.23, shuffle=True, random_state=66)

# x_train=np.array(range(11))
# y_train=np.array(range(11))
# x_test=np.array([11,12,13])
# y_test=np.array([11,12,13])
# x_val = np.array([14,15,16])
# y_val = np.array([14,15,16])

#2. 모델구성
model = Sequential()
model.add(Dense(5, input_dim=1))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(2))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=1, 
#          validation_data=(x_val, y_val), verbose=1)
           validation_split=0.3)

#4. 평가,예측
loss=model.evaluate(x_test, y_test)
print('loss: ', loss)

y_predict = model.predict([17])
print("17의 예측값:", y_predict)

# result
# loss:  1.658227847656235e-05
# 17의 예측값: [[16.992413]]
