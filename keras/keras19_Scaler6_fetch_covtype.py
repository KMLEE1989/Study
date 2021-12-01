from tensorflow.python.keras.backend import softmax
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

#데이터셋을 불러오고 그다음 데이터의 분석을 위해 데이터를 출력해봅니다.
datasets=fetch_covtype()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape) #->구해 보았더니 (581012, 54) (581012,) 이렇게 나오네요
print(y)#->feature를 구해 보도록 하구요
print(np.unique(y)) #->[1 2 3 4 5 6 7] 이렇게 나옵니다.  

y=to_categorical(y)
print(y)
print(y.shape) #(581012, 8)

#다음은 train test split 과정으로 넘어갑니다. 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#그리고 우리는 train과 test의 값중 test 값을 사용할 거랍니다. 
print(x_train.shape, y_train.shape) #->(464809, 54) (464809, 8)
print(x_test.shape, y_test.shape) #-> (116203, 54) (116203, 8)

model = Sequential() 
model.add(Dense(10, activation='linear', input_dim=54))  #(116203, 54) 
model.add(Dense(10))                                               #54는 input의 노드가 되구요
model.add(Dense(70, activation='relu'))
model.add(Dense(60))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(8, activation='softmax'))  

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=80, validation_split=0.2, callbacks=[es])

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:10])
print(y_test[:10])
print(results)

#Activation Relu(미적용)
# loss : 0.644293487071991 - accuracy: 0.7220553755760193

# MinMax
# loss: 0.6359293460845947 - accuracy: 0.7218058109283447

# Standard
# loss: 0.6333673596382141 - accuracy: 0.7224684357643127

# Robust
# loss: 0.6315158009529114 - accuracy: 0.7238194942474365

# MaxAbs
# loss: 0.6353898644447327 - accuracy: 0.7237765192985535

##############################################################

#Activation Relu(적용)

#결과
# loss:  0.3777937591075897 - accuracy: 0.8439111113548279

# MinMax
# loss: loss: 0.2937 - accuracy: 0.8809

# Standard
# loss: loss : 0.28226804733276367 - accuracy: 0.8864831328392029

#Robust
# loss: 0.2688366770744324 - accuracy:0.8924984931945801

#MaxAbs
# loss : 0.30900809168815613 - accuracy: 0.8733423352241516

