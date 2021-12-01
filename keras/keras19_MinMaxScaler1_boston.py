###########################################################
#각각의 Scaler의 특성과 정의 정리해놓을것! 
###########################################################
#1.standardscaler : 기본스케일의 평균과 표준편차 사용 -> 이상치가 있는 경우 균형 잡힌 척도를 보장할 수 없다
#2. MinMaxScaler: 최대/최소값이 각각 1,0이 되도록 스케일링 -> 모든 column 값이 0~1사이에 있도록 데이터를 재조정한다. 다만 이상치가 있는 경우 변환된 값이 매우 좁은 범위로 압축될 수 있다. 
#3. MaxAbsScaler: 최대절대값과 0이 각각 1,0이 되도록 스케일링 -> 절대값이 0~1사이에 있도록 잡아준다. 즉 -1~1사이로 재조정하는데 양수 데이터로만 구성된 특징 이있다. 그리고 큰 이상치에 민감할 수 있다.
#4. RobustScaler: 중앙값(median)과 IQR(interquatile Range)사용. 아웃라이어의 영향을 최소화 스탠다드 스케일과 유사하지만 양 끝 25퍼센트와 75퍼센트의 값들을 다룬다. 
#기본적으로 standardscaler와 RobustScaler의 변환된 결과가 유사 형태로 반환된다. 
#민맥스케일링 같은 경우 0~1사이에 있도록 데이터를 잡아주지만 한쪽으로 쏠림 현상이 있거나 다른 아웃라이어가 있어도 거의 분포 형태가 비슷하게 유지된채 범위 값이 조절된다.
# y= x에대한 데이터 타겟이므로 train에 영향을 미치지도 않는다. x1,x2 각 칼럼 마다 따로 전처리를 하며 0과 1사이의 데이터가 아닐경우 그대로 쓴다. test는 train의 비율에 맞추어 조정해 준다. 
#######

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np
from tensorflow.python.keras.backend import sigmoid

datasets=load_boston()
x=datasets.data
y=datasets.target

#print(np.min(x), np.max(x))  # 0.0 711.0
# x = x/711.               #부동소수점으로 나눈다 
# x = x/np.max(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
# scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#2. 모델구성
model = Sequential() 
model.add(Dense(50, input_dim=13))
model.add(Dense(40))
model.add(Dense(30))
model.add(Dense(20, activation='relu'))
model.add(Dense(10))
model.add(Dense(8, activation='relu'))
model.add(Dense(6))
model.add(Dense(5))
model.add(Dense(4))
model.add(Dense(3))
model.add(Dense(1))

#3.

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train, y_train, epochs=1000, batch_size=13, validation_split=0.5)  #로스와 발로스 반환

#4.
loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)

'''

'''  #relu(미적용)
#결과
#loss : 21.101839065551758
#r2스코어: 0.7445826751947613

# MinMax
# loss : 19.388029098510742
# r2스코어: 0.7653266861476019


# Standard
#loss : 19.016672134399414
#r2스코어: 0.7698216027206496


#Robust
#loss : 20.277883529663086
#r2스코어: 0.7545558677080965


#MaxAbs
#loss : 21.733667373657227
#r2스코어: 0.7369350039303418

######################################################      
#Activation Relu(적용)

#결과
# loss : 18.928192138671875   (Relu 적용 향상)
# r2스코어: 0.7708925639216612 

# MinMax
# loss :  19.430370330810547
# r2스코어: 0.7648141977130958


# Standard
# loss : 28.5865535736084
# r2스코어: 0.6539874418229767

#Robust
# loss : 14.592887878417969
# r2스코어: 0.8233672275764976


#MaxAbs
# loss : 19.17517852783203
# r2스코어: 0.7679030395783546




