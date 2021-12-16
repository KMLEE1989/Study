from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, Activation,MaxPooling2D
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler,RobustScaler,MaxAbsScaler
from sklearn.metrics import r2_score
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

datasets = load_diabetes()
x = datasets.data           # 원본데이터는 나중에 어떻게 쓸지 모르니까  그냥 둠
y = datasets.target 

print(x.shape) # x형태  (442, 10)    -> 칼럼하나 줄여서 12 만들어서 (506,2,2,3) 4차원 형태로 conv2d로 받아서 cnn모델링가능
print(y.shape) # y형태  (442,)
#print(datasets.feature_names) # 컬럼,열의 이름들
#print(datasets.DESCR) # 데이터셋 및 컬럼에 대한 설명

xx = pd.DataFrame(x, columns=datasets.feature_names)
print(type(xx))        
print(xx) 

#xx[''] = y 
print(xx) 
print(xx.corr())      

# import matplotlib.pyplot as plt
# import seaborn as sns   
# plt.figure(figsize=(10,10))
# sns.heatmap(data=xx.corr(), square=True, annot=True, cbar=True)
# # seaborn heatmap 개념정리
# plt.show()
#xx= xx.drop(['CHAS','price'], axis=1)  
xx = xx.values 

x_train,x_test,y_train,y_test = train_test_split(xx,y, train_size=0.8, shuffle=True, random_state=49)

print(x_train.shape,y_train.shape)      #(455, 12) (455,)
print(x_test.shape,y_test.shape)        #(51, 12) (51,)

scaler =MinMaxScaler() 
x_train = scaler.fit_transform(x_train).reshape(len(x_train),5,2,1)
x_test = scaler.transform(x_test).reshape(len(x_test),5,2,1)

model = Sequential()
model.add(Conv2D(10,kernel_size=(2,2),strides=1,padding='same', input_shape=(5,2,1), activation='relu'))    # 2,2,10                                                                           # 1,1,10
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
#mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True, filepath=f'./_ModelCheckPoint/keras35_1_boston{krtime}.hdf5')
model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])#,mcp


#4. 평가 예측

loss = model.evaluate(x_test,y_test)
print('loss : ', loss)

y_predict = model.predict(x_test)

r2 = r2_score(y_test,y_predict) 
print('r2스코어 : ', r2)


# loss :  2485.963134765625
# r2스코어 :  0.5334426614831349