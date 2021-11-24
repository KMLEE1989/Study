import numpy as np 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.tools.docs.doc_controls import T 

x=np.array([[1,2,3,4,5,6,7,8,9,10],
           [1,1.1,1.2,1.3,1.4,1.5,
            1.6,1.5,1.4,1.3],
           [10,9,8,7,6,5,4,3,2,1]])
x=np.transpose(x)
y=np.array([11, 12, 13, 14, 15, 16, 17, 18, 19, 20])


#print(x.shape)
#print(y.shape)


model = Sequential()   
model.add(Dense(2, input_dim=3))  #바둑돌 3으로 시작  앞에 숫자는 상관 없음 
model.add(Dense(30))
model.add(Dense(50))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(80))
model.add(Dense(30))
model.add(Dense(1))    #하이퍼튜닝 안정화 

#3.컴파일, 훈련
model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=1000, batch_size=1)


#4. 평가,예측
loss=model.evaluate(x,y)
print('loss :', loss)
y_predict=model.predict([[10, 1.3,1]])
print('[10,1.3,1]의 예측값:', y_predict)


#시작!!
#[[10,1.3,1]] 결과값 예측
#결과는 20 +- 0.5 에 맞추어야 해
#result loss : 0.000355
#              예측값: [[19.980318]]
#다층 퍼셉트론(multi-layer perceptron, MLP)는 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태입니다.