#이제 다중분류로 가보도록 해봅시다.
#세개 이상의 선택지를 고르는 것을 다중 분류라고 정의한다. 
#다중분류는 output 부분의 activation에 sigmoid가 아닌 softmax를 사용하는데
#위 이진 분류의  binary cross Entropy 대신에 categoical_cross entropy를 사용한다.
# wine case에 원핫 인코딩과 원핫 인코딩의 원리 목적을 설명할것.

import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default

datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

#print(x.shape, y.shape)  #(150, 4) (150,)
#print(y)
#print(np.unique(y)) #[0 1 2]  라벨값이란?  여기서는 4래 여기서 (150,4) 그리고 (150,3) 으로 만들자! 원핫 인코딩을 이용해!
y=to_categorical(y)
print(y)
print(y.shape)  #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential() 
model.add(Dense(10, activation='linear', input_dim=4))  #(30, 4) (30, 3)
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

#결과
#loss : 0.1266656219959259
#accuracy: 0.9666666388511658







