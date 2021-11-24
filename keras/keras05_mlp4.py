import numpy as np
from numpy.core.fromnumeric import transpose 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense

#1. 데이터 
x= np.array([range(10)])
print(x)
x=np.transpose(x)
print(x.shape)   #(10,1)->이어도 demension 1로 잡는다 

y=np.array([[1,2,3,4,5,6,7,8,9,10],
           [1,1.1,1.2,1.3,1.4,1.5,
            1.6,1.5,1.4,1.3],
           [10,9,8,7,6,5,4,3,2,1]])
y=np.transpose(y)

print(y.shape) #(10,3)


model = Sequential()   
model.add(Dense(10, input_dim=1))
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(250))
model.add(Dense(190))
model.add(Dense(160))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(3))  

model.compile(loss='mse', optimizer='adam')  
model.fit(x, y, epochs=3000, batch_size=1)


loss=model.evaluate(x,y)
print('loss :', loss)
y_predict=model.predict([[9]])
print('[9]의 예측값:', y_predict)

#10,1.3,1
#Result 
#loss : 0.00703
#[9]의 예측값: [[9.992932   1.522013   0.99503803]



#(9) (10,1.3,1)
