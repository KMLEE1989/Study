#relu만 사용 하였을때 큰폭으로 성능이 향상 되었다. 모든 데이터스케일링이 성공적이었으며 대부분 절반이상의 cost가 감소 되었다. 


from tensorflow.python.keras.backend import softmax
import numpy as np
from sklearn.datasets import fetch_covtype
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model 
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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

input1=Input(shape=(54,))
dense1=Dense(10)(input1) 
dense2=Dense(10)(dense1)                                               #54는 input의 노드가 되구요
dense3=Dense(70, activation='relu')(dense2)
dense4=Dense(60)(dense3)
dense5=Dense(50, activation='relu')(dense4)
dense6=Dense(10)(dense5)
output1=Dense(8, activation='softmax')(dense6)
model=Model(inputs=input1, outputs=output1)

#model.save("./_save/keras24_hamsu1_fetch_covtype_load_save.practice.h5")
#model=load_model("./_save/keras24_hamsu1_fetch_covtype_load_save.practice.h5")


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)

model.fit(x_train, y_train, epochs=100, batch_size=80, validation_split=0.2, callbacks=[es])

model.save("./_save/keras24_hamsu1_fetch_covtype_load_save.practice.h5")

loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:10])
print(y_test[:10])
print(results)

#result
# loss : 0.2953770160675049
# accuracy: 0.8819565773010254
