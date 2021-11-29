#2021/11/29 수업의 의도 
#분류 작업 시작하기 이진분류와 다중분류는 분류의 대표적인 모델
#이진분류-> 회귀와 함께 가장 기초적인 분석 방법. 데이터가 어느 범주(category)에 속하는지를 판단
#이진 분류의 특성 sigmod 함수를 사용한다.  *이진분류는 다중분류에 속한다.
#사실 레이어 연산에서 부동소수점 연산이 훨씬 잘 구동 된다는 것을 인지하도록 하자.
# y=wx+b의 output을 위한 디폴트 함수는 보이지 않지만 linear이다.
#레이어에서 지속적으로 데이터의 증폭과 정확도 missing을 방지하는 방법으로 
#기본적인 디폴트 아웃풋 linear-> sigmoid로 마지막 아웃풋 노드에서 바꿔 준다.
#여기서 sigmoid는 은닉 레이어에 들어갈 수 있고 0과 1사이의 값만 처리한다.
#정수가 아닌 실수로의 값을 대표하는게 sigmoid
#sigmoid와 set로 움직이는 것은  "컴파일 훈련 부분 로스 함수 정의 부분 [mse]=mean squaded error 를 binary cross Entropy로 바꿔준다"
#그리고 회기 모델의 보조 기표 r 스퀘어는 당연이 분류 모델에서는 사용 되지 않는다.(주의)
#같은 compile열에서 metrics에 acuuracy가 사용 되는데 accuracy는 실제 데이터에서 예측 데이터가 얼마나 같은지를 판단하는 단순 지표이다.
#그러므로 에큐려시는 training에 영향을 미치지 않는다. 
#evaluate 의 첫번째 loss는 분류 모델에서 고정값으로 사용된다. 너무 많은  sigmoid를 사용하게 되면 acuuracy 작아지능 성질 0과 1의 사이로만 측정이 되어버리기 때문에 
#사용할때 주의 하는 것이 좋다. 

import numpy as np
from sklearn import datasets 
from sklearn.datasets import load_breast_cancer
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt 

#1. 데이터

datasets = load_breast_cancer()
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
#print(datasets)
print(datasets.DESCR)
print(datasets.feature_names)

x = datasets.data
y = datasets.target
#print(x.shape,y.shape) #(569, 30) (569,)


print(y)
#print(y[:10]) 0과 1을 본 순간 2진분류 파악 -> 시그모이드 -> 바이너리 엔트로피  

print(np.unique(y))  #[0 1]

#2. 모델구성 

x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66)

model = Sequential() 
model.add(Dense(100, activation='linear', input_dim=30))
model.add(Dense(80))
model.add(Dense(70, activation='linear'))
model.add(Dense(60))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='sigmoid'))

#3. 컴파일, 훈련 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

start=time.time()
model.fit(x_train, y_train, epochs=1000, batch_size=1, validation_split=0.2, callbacks=[es])  #얼리스타핑을 콜백으로 땡기겠다 리스트
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)
y_predict= model.predict(x_test)
print(y_predict)



#이벨류에잇의 첫번째 로스는 고정값 절대 고정 
#loss : [0.6710209250450134, 0.6198830604553223] 첫번째 바이너리 엔트로피 진짜 로스 두번째는 매프릭스(에큐러시는 영향을 안끼친다. 메트릭스 현재 훈련의 상황만 보여준다 평가지표에 의한), 두개 이상 리스트
'''
y_predict= model.predict(x_test)


# from sklearn.metrics import r2_score
# r2=r2_score(y_test, y_predict)
# print('r2스코어:', r2) 이건 회귀지표에만 쓰이기 때문에 분류 지표에서는 의미가 없다  로스는 동일로 사용  

plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()

'''

#result : loss: 0.1855 - accuracy: 0.9298  너무 많은  sigmoid를 사용하게 되면 acuuracy 작아지능 성질 0과 1의 사이로만 측정이 되어버리기 때문에 