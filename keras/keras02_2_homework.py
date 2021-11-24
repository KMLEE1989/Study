#import tensorflow as tf
from tensorflow.keras.models import Sequential  #(tensorflow의 keras의 순차모델을 쓸거야) 
from tensorflow.python.keras.layers import Dense 
import numpy as np
#from tensorflow.python.keras.models import Sequential 


#1.데이터 (전처리 데이터 정제된:모델에 적합한 데이터)
x = np.array([1,2,3])  #np열
y = np.array([1,2,3])
#요 데이터를 훈련해서 최소의 loss를 만들어보자. 

#2. 모델 구성
model = Sequential()   #S는 대문자 그래서 클래서    시퀀셜이라는 클래스를 모델로 정의 하겠다
model.add(Dense(5, input_dim=1)) #(한개의 노드가 들어간다 인풋딤) 한개의 차원 노드는 아웃풋 한줄이 하나의 레이어  괄호 앞에 아웃풋 뒤가 인풋
model.add(Dense(50))  #시퀀셜이기 때문에 인풋을 쓸 필요가 없어
model.add(Dense(80))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(180))
model.add(Dense(90))
model.add(Dense(1))  #인공신경망 완성  (아웃풋)

#3. 컴파일,훈련
model.compile(loss='mse', optimizer='adam')  #그은선과 실제 데이터의 거리 loss  민스쿼드에라 평균 제곱(하는 이유 양수화 음수가 나올수 없다) 에라 (작으면 좋다)    옵티마이저(엠에스이를 최소화 )

model.fit(x, y, epochs=30, batch_size=1) #핏은 훈련을 시키다  에포 몇번 선을 긋겠다   배치가 작을수록 훈련은 잘돼 (속도차이) (#3이 끝나면 최적의 웨이트를 만들 준비 ) 


#4. 평가,예측
loss = model.evaluate(x,y)  #최소의 로스가 될 준비 
print('loss: ',loss)
result=model.predict([4]) 
print('4의 예측값: ', result)


#케라스 1-1 1-2 는 단일 신경망 복합 신경망은 성능이 더 좋다!
#인풋레이어 아웃풋레이어 가운데(바디) 신경망들은 히든 레이어 (바디를 계속 고치는거 , 에포, 배치 도 하이퍼 파라미터 튜닝) 
#에포를 50으로 고정 했을 경우 4.0000000, 3.9999999가 나오도록  하이퍼 파라미터 튜닝을 해서

