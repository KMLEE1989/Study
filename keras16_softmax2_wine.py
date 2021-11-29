#ONE HOT encoding의 목적: 인간과 컴퓨터는 데이터를 바라보는 형태가 다르다. 가장 기본적인 개념은
#컴퓨터가 읽을 수 있도록 데이터를 0과 한개의1의 값으로 데이터를 구별하는 인코딩이다.
#기본적인 구성은 0으로 이루어진 벡터에 단 한개의 1의 값으로 해당하는 할당량으로 데이터 변환을 해주는 함수이다.

#Food name  Categorical number  calories
#APPLE         1                 95
#Chicken       2                 231
#Broccoli      3                 50

# 이경우 컴퓨터는 1,2,3 중 카테고리칼 넘버에서 비중이 제일 높은 3에 가중치를 두고 연산을 하는 것이 맞다.
# 하지만 잊지 말하야 할 것은 사과 치킨 브로콜리의 중요도는 사실 동등하기 때문에 
#Apple   Chicken  Broccoli  calories
#  1        0       0          95
#  0        1       0         231
#  0        0       1           50
# 이렇게 나타난다. 그러면 세 음식의 중요도는 같이 연산된다. 

#이경우 사실상 activation node의 softmax 부분의 비율로 따졌을때 1이 되는 것이 맞다.
#그러면 y의 노드가 칼럼 피쳐로 바귀고 y=(N,3)로 자연스럽게 코딩 된다.

import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape)  #(178, 13) (178,)
print(y)
print(np.unique(y)) #[0 1 2]  라벨값이란?  여기서는 4래 여기서 (150,4) 그리고 (150,3) 으로 만들자! 원핫 인코딩을 이용해!


y=to_categorical(y)
print(y)
print(y.shape)  #(178, 3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential() 
model.add(Dense(10, activation='linear', input_dim=13))  #(36, 13) (36, 3)
model.add(Dense(10))
model.add(Dense(70, activation='linear'))
model.add(Dense(60))
model.add(Dense(50, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))

#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)


model.fit(x_train, y_train, epochs=100, batch_size=1, validation_split=0.2, callbacks=[es])  #얼리스타핑을 콜백으로 땡기겠다 리스트



loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:5])
print(y_test[:5])
print(results)

#Result
# loss : 0.249056875705719
# accuracy: 0.9166666865348816
