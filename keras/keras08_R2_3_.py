##R2가 먼지 찾아라!!!
##R2란 회귀선에 각각의 값들이 얼마나 가까운지를 측정, 회귀선이 얼마나 실제값을 잘 예측할 수 있는 가
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
#from sklearn.metrics import r2_score

#1. 데이터 
x = np.array([1,2,3,4,5])
y = np.array([1,2,4,3,5])

x_train, x_test, y_train, y_test= train_test_split(x,y,
                train_size=0.7, shuffle=True, random_state=66)

#2. 모델구성
model = Sequential() 
model.add(Dense(20, input_dim=1))
model.add(Dense(60))
model.add(Dense(100))
model.add(Dense(130))
model.add(Dense(70))
model.add(Dense(40))
model.add(Dense(1))

#3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x,y, epochs=1100, batch_size=1)

#4. 평가, 예측
loss=model.evaluate(x_test, y_test)  #왜 로스값이 최종값인가 
print('loss :', loss)

y_predict= model.predict(x)

from sklearn.metrics import r2_score
r2=r2_score(y, y_predict)
print('r2스코어:', r2)

#loss : 0.28018006682395935
#r2스코어: 0.8022155558565871



# first
# loss : 21.049678802490234
# r2스코어: -1.0100495337350797

'''
plt.scatter(x, y)
plt.plot(x, y_predict, color='red')
plt.show()
'''
#second
# loss : 10.532241821289062
# r2스코어: 0.2594521087821883

