#relu를 사용하였을때 데이터 성능 향상이 뚜렷했으며 특히 로스 감소와 accuracy의 증폭값이 매우 컸다. Robustscaler를 사용했을때 두렷한 성능향상을 보였다. 


import numpy as np
from sklearn.datasets import load_iris
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential,load_model 
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.utils.generic_utils import default
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input

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

#scaler = MinMaxScaler()
#scaler=StandardScaler()
scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


#model=load_model("./_save/keras23_hamsu1_iris_load_save.practice.h5")
# input1 = Input(shape=(4,)) 
# dense1=Dense(10)(input1)
# dense2=Dense(10)(dense1)
# dense3=Dense(70,activation='relu')(dense2)
# dense4=Dense(60)(dense3)
# dense5=Dense(50, activation='relu')(dense4)
# dense6=Dense(10)(dense5)
# output1=Dense(3)(dense6)
# model=Model(inputs=input1, outputs=output1)

#model.save("./_save/keras23_hamsu1_iris_load_save.practice.h5")



#3. 컴파일, 훈련 
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# from tensorflow.keras.callbacks import EarlyStopping
# es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)


# model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  #얼리스타핑을 콜백으로 땡기겠다 리스트

model=load_model("./_save/keras23_hamsu1_iris_load_save.practice.h5")


loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])

results=model.predict(x_test[:5])
print(y_test[:5])
print(results)

# results
# loss : 0.5372698903083801
# accuracy: 0.6333333253860474


