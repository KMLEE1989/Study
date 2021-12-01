###########################################################
#각각의 Scaler의 특성과 정의 정리해놓을것! 
###########################################################

#######

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
import numpy as np

datasets=load_diabetes()
x=datasets.data
y=datasets.target


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=66)

#scaler = MinMaxScaler()
#scaler=StandardScaler()
#scaler=RobustScaler()
#scaler=MaxAbsScaler()
#scaler.fit(x_train)
#x_train=scaler.transform(x_train)
#x_test=scaler.transform(x_test)


model = Sequential() 
model.add(Dense(50, input_dim=10))
model.add(Dense(70))
model.add(Dense(100))
model.add(Dense(120, activation='relu'))
model.add(Dense(180))
model.add(Dense(200, activation='relu'))
model.add(Dense(170))
model.add(Dense(120))
model.add(Dense(80))
model.add(Dense(40))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

from tensorflow.keras.callbacks import EarlyStopping

es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1)

hist = model.fit(x_train, y_train, epochs=1000, batch_size=8, validation_split=0.5, callbacks=[es])  #로스와 발로스 반환

loss=model.evaluate(x_test, y_test) 
print('loss :', loss)

y_predict= model.predict(x_test)

from sklearn.metrics import r2_score
r2=r2_score(y_test, y_predict)
print('r2스코어:', r2)


#결과 Activation Relu(미적용)
# loss : 4039.931396484375
# r2스코어: 0.3515744391854546

# MinMax
# loss : 3828.86865234375
# r2스코어: 0.3854508311503636

# Standard
# loss : 3204.404052734375
# r2스코어: 0.48567997060051793


#Robust
# loss : 3153.81201171875
# r2스코어: 0.49380026685070977


#MaxAbs
# loss : 3156.894775390625
# r2스코어: 0.4933054134384872

######################################################      
#Activation Relu(적용)

#결과
# loss : 3804.687744140625
# r2스코어: 0.3893320372619743

# MinMax
# loss : 3673.18212890625
# r2스코어: 0.41043915634786576


# Standard
# loss : 4567.4443359375
# r2스코어: 0.26690637081395347

#Robust
# loss : 4485.875
# r2스코어: 0.2799986723314707

#MaxAbs
# loss : 4361.18798828125
# r2스코어: 0.30001140711298935