import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

#1.데이터
x= np.transpose([[1,2,3,4,5,6,7,8,9,10],            #행과 열을 바꾸기 위해 utilize transpose method 
            [1,1.1,1.2,1.3,1.4,1.5,
             1.6,1.5,1.4,1.3]])
y= np.array([11,12,13,14,15,16,17,18,19,20])

print(x.shape)
print(y.shape)


#2.모델구성
model = Sequential()   
model.add(Dense(2, input_dim=2))
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))  



#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=1000, batch_size=1)


#4. 평가,예측
loss=model.evaluate(x,y)
print('loss :', loss)
y_predict=model.predict([[10, 1.3]])
print('[10,1.3]의 예측값:', y_predict)


#result :0.00014
#        predict 20.015

#Main mission: matrix x에서 10,2 를 설정 했을 경우 y값에서의 충돌이 생기므로  x를 transpose를 사용하여 행열을  바꿔준다 


#x=np.transpose 가로세로가 바뀌고 행열이 틀어 진다 
#x=x.T 
# 프레딕트에 들어가는 행열은 input 디멘션에 들어가는것과 같다 
# 열 우선 행 무시