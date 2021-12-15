import numpy as np
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import train_test_split
import time
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 

a=np.array(range(1,101))
x_predict= np.array(range(96,106))    #.reshape(1,10,1)
print(x_predict)

size= 5  # x4개, y1개

def split_x(dataset, size):
    aaa = []
    for i in range(len(dataset)-size) :
        aaa=[]
        for i in range(len(dataset)-size+1):
            subset = dataset[i : (i+size)]
            aaa.append(subset)
        return np.array(aaa)

bbb=split_x(a,size)
print(bbb)

#print(bbb.shape) (96, 5)

x=bbb[:, :-1].reshape(96,4,1)
y=bbb[:, -1]

print(x.shape) #(96, 4,1)
print(y.shape)  #(96,)


pred=split_x(x_predict,5)
#print(pred.shape) (6,5)
x_pred=pred[:, :4].reshape(6,4,1)

x_train, x_test, y_train, y_test= train_test_split(x,y, train_size=0.7, shuffle=True, random_state=1)

#모델구성하시오

model = Sequential()
model.add(LSTM(100, activation='linear', input_shape=(4,1)))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1))


# print(bbb.shape) #(6,5)

# x=bbb[:, :4]
# y=bbb[:, 4]
# print(x,y)


model.compile(loss='mae', optimizer='adam')

es = EarlyStopping(monitor="val_loss", patience=50, mode='min',verbose=1,baseline=None, restore_best_weights=True)

start= time.time()

model.fit(x_train,y_train,epochs=1000, batch_size=10,validation_split=0.2, callbacks=[es])


end=time.time()-start


# #4. 평가 예측
loss=model.evaluate(x_test,y_test)
# # y=np.array([5,6,7]).reshape(1,3,1)
result=model.predict(x_pred)
#result = model.predict([[[50],[60],[70]]])
print(result)
print("걸린시간 :" , round(end, 2), '초 ')


# [[ 99.564835]
#  [100.55266 ]
#  [101.5404  ]
#  [102.528015]
#  [103.51556 ]
#  [104.50301 ]]