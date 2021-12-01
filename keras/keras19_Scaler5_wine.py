import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split

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
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)


print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


model = Sequential() 
model.add(Dense(50, activation='linear', input_dim=13)) 
model.add(Dense(55))
model.add(Dense(60, activation='relu'))
model.add(Dense(50))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='linear'))
model.add(Dense(3, activation='softmax'))

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

#Result Activation Relu(미적용)
# loss : 0.249056875705719
# accuracy: 0.9166666865348816

# MinMax
# loss: 0.4188 - accuracy: 0.9722

# Standard
# loss: 0.3493 - accuracy: 0.9722

# Robust
# loss: 0.4725 - accuracy: 0.9722

# MaxAbs
# loss: 0.1563 - accuracy: 0.9722

###############################################

#Activation Relu(적용)

#결과
# loss: 0.1859 - accuracy: 0.9444

# MinMax
# loss: 0.3068 - accuracy: 0.9722

# Standard
# loss: 0.1460 - accuracy: 0.9722

#Robust
# loss: 0.1452 - accuracy: 0.9722

#MaxAbs
# loss: 0.0628 - accuracy: 1.0000