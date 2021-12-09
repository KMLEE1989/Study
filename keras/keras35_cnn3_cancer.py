from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_breast_cancer()
x = datasets.data           
y = datasets.target 

print(x.shape) # x형태  (569, 30)
print(y.shape)       #  (569,)
#print(datasets.feature_names) # 컬럼,열의 이름들
#print(datasets.DESCR)

xx = pd.DataFrame(x, columns=datasets.feature_names)
print(type(xx))        
print(xx) 

print(xx) 
print(xx.corr()) 

xx = xx.values 

x_train,x_test,y_train,y_test = train_test_split(xx,y, train_size=0.8, shuffle=True, random_state=49)

print(x_train.shape,y_train.shape) #(455, 30) (455,)
print(x_test.shape,y_test.shape)   #(114, 30) (114,)

scaler =MinMaxScaler() 
x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,6,1)
x_test = scaler.transform(x_test).reshape(len(x_test),5,6,1)

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(5,6,1), activation='relu'))    # 2,2,10                                                                           # 1,1,10
model.add(Conv2D(10,kernel_size=(2,2), strides=1, padding='same', activation='relu'))                       # 2,2,10 
model.add(MaxPooling2D(2,2))                                                                                # 1,1,10     
model.add(Flatten())       
model.add(Dense(64))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)

model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])#,mcp


#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)


# loss :  0.03592715039849281
# r2스코어 :  0.8439995854613449