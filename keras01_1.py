#import tensorflow as tf
from tensorflow.keras.models import Sequential  #(tensorflow의 keras의 순차모델을 쓸거야) 
from tensorflow.python.keras.layers import Dense 
import numpy as np
#from tensorflow.python.keras.models import Sequential 





#1.데이터 
x = np.array([1,2,3])  #np열
y = np.array([1,2,3])


#2. 모델 구성
model = Sequential()   #S는 대문자 그래서 클래서    시퀀셜이라는 클래스를 모델로 정의 하겠다
model.add(Dense(1, input_dim=1))


#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')  #그은선과 실제 데이터의 거리 loss  민스쿼드에라 평균 제곱(하는 이유 양수화 음수가 나올수 없다) 에라 (작으면 좋다)    옵티마이저(엠에스이를 최소화 )

model.fit(x, y, epochs=2000, batch_size=1) #핏은 훈련을 시키다  에포 몇번 선을 긋겠다   배치가 작을수록 훈련은 잘돼 (속도차이) (#3이 끝나면 최적의 웨이트를 만들 준비 )


#4. 평가,예측
loss = model.evaluate(x,y)  #최소의 로스가 될 준비 
print('loss: ',loss)
result=model.predict([4]) 
print('4의 예측값: ', result)



'''
#loss:  0.00032318069133907557
#4의 예측값:  [[3.9620025]]
'''