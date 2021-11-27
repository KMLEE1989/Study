#과제 의도: Earlystopping 에 대한 이해.
#overfitting은 training data에만 지나치게 적응되어서 그 외의 데이터에는 제대로 대응하지 못하는 상태를 말한다. 
#patience 값의 문제점 파악하기 -> 전체적인 훈련(epoch)구간의 최소값에서 멈춘건지 patience 구간에서만 비교하여 멈춘건지를 알아야 한다.
#loss는 결과 값의 차이를 의미 -> 값이 작을 수록 좋다. 0.0000에 수렵할수록 좋은 모델임을 의미 
#그러면 머신 러닝의 궁극적인 목표를 알아 보자-> 궁극적인 목표는 training dataset을 이용하여 학습한 모델을 가지고 test dataset을 예측 하는 것이다.
#training dataset과 test dataset이 정확히 일치한다면, training dataset에 fitting될 수록 모델의 예측 정확도는 증가. 그러나 문제는 training dataset과 test dataset은 조금씩 다른 경향을 보인다.  
#이때 test dataset은 학습 과정에서 참조할 수 없음. 결론적으로 머신 러닝의 모델은 training dataset만을 가지고 test dataset을 잘 예측하도록 합습되어야 한다.
# y를 LOSS로 정의하고 X를 Epoch라고 가정한 함수 선형을 그린 경우 training loss와 test loss가 같이 감소하는 구간을 ->underfitting
# 위의 그래프에서 training loss는 감소하지만 test loss가 증가하는 구간을 overfitting이라고 정의한다.
#우리의 목적: 학습을 통해 머신 러닝 모델의 underfitting된 부분을 제거하면서 overfitting이 발생하기 직전 학습을 멈추는 것인데 이를 위해 머신 러닝에서
#validation set을 이용한다. 
#Validation  loss가 증가하는 시점부터 overfitting이 발생했다고 판단하고, 이에 따라 학습을 중단한다. 

'''
#callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',patience=5),keras.callbacks.ModelCheckpoint(filepath='best_model.h5', monitor='val_loss',save_best_only=True)]
#keras_model.fit(x_train,y_train, barch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=callbacks)
#Keras_model_best=keras.models.load_model('best_model.h5')

#validation dataset또한 test dataset을 완벽히 표현하지는 못 하기 떄문에 validation loss가 최소가 되는 시점이 test loss가 최소가 되는 시점과
#정확히 일치하지는 않을 수도 있다. 
#Step: 
#Step (1): 주어진 dataset을 training, validation, test dataset들로 나눈다. 일반적으로 각 dataset의 비율은 A:B:B으로 설정한다.
# Step (2): Training dataset을 이용하여 모델을 학습시킨다.
# Step (3): Validation dataset을 이용하여 모델의 정확도 및 validation dataset에 대한 loss를 계산한다.
# Step (4): 만약 validation loss가 증가했다면 학습을 종료한다. 그렇지 않을 경우에는 (2)로 돌아가서 학습을 계속 진행한다.
# Step (5): Test dataset을 이용하여 모델의 최종 정확도를 평가한다.

# restore_best_weights
#Early stopping으로 일정 patience 으로 연속적인 훈련 후 값이 향상되지
#않을 경우 종료시킬 것인지를 나타냄. 
#이경우 restore_best_weights=True를 사용할 경우 epoch중에서 최적의 값으로 모델 복구를 도와준다. 

맨밑 주석으로 이 코딩에 대한 결과
'''

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np 
#import matplotlib.pyplot as plt 
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import time

datasets = load_diabetes()
x=datasets.data
y=datasets.target


print(x)
print(y)
print(x.shape)
print(y.shape)

print(datasets.feature_names)
print(datasets.DESCR)


x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.7, shuffle=True, random_state=49)

model = Sequential() 
model.add(Dense(50, input_dim=10))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(120))
model.add(Dense(180))
model.add(Dense(200))
model.add(Dense(170))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

start=time.time()
hist = model.fit(x_train, y_train, epochs=10000, batch_size=8, validation_split=0.5, callbacks=[es])  
#hist 정의 Earlystopping 
end=time.time() - start

print("걸린시간:  ", round(end,3))

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)


print("=======================================================================================")
print(hist)
print("=======================================================================================")
print(hist.history)
print("=======================================================================================")
print(hist.history['loss'])
print("=======================================================================================")
print(hist.history['val_loss'])
print("=======================================================================================")




import matplotlib.pyplot as plt 


plt.figure(figsize=(9,5))
plt.plot(hist.history['loss'],marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'],marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()



# restore_best_weights

#Val_loss의 전체 갯수: 74
#val_loss의 최소: 3192.65087890625 23번째 
#stop epoch의 val_loss: 3242.452880859375 
#Result : Stop epoch의 val_loss가 제일 낮은 값이 아님 (Early stop 구간)
#val_loss 전체 갯수 74-50=24    24-1 번째 Val_loss가 제일 작습니다. 3192.65087890625 

