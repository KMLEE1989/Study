
import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input


datasets = load_wine()
print(datasets.DESCR)
print(datasets.feature_names)

x=datasets.data
y=datasets.target

print(x.shape, y.shape)  
print(np.unique(y)) 


y=to_categorical(y)
print(y)
print(y.shape)  #(178, 3)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

input1 = Input(shape=(13,)) 
dense1=Dense(10)(input1) 
dense2=Dense(10)(dense1)
dense3=Dense(70,activation='relu')(dense2)
dense4=Dense(60)(dense3)
dense5=Dense(50)(dense4)
dense6=Dense(50)(dense5)
dense7=Dense(10)(dense6)
output1=Dense(3, activation='softmax')(dense7)
model=Model(inputs=input1, outputs=output1)


#3. 컴파일, 훈련 
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
es=EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)


model.fit(x_train, y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[es])  



loss=model.evaluate(x_test, y_test) 
print('loss :', loss[0])
print('accuracy:' ,loss[1])
results=model.predict(x_test[:5])
print(y_test[:5])
print(results)

# results
# loss : 0.7908433675765991
# accuracy: 0.6944444179534912