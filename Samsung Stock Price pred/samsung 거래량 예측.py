import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM

path = "../_data/SamsungK/"

df1=pd.read_csv(path+'삼성전자.csv',thousands=',')

df2=pd.read_csv(path+'키움증권.csv',thousands=',')

# print(df1) #[1120 rows x 17 columns]
# print(df1.shape) #(1120, 17)

# print(df2) #[1060 rows x 17 columns]
# print(df2.shape) #(1060, 17)

# print(df1.info) #[1120 rows x 17 columns]
# print(df2.info) #[1060 rows x 17 columns]
df1=df1.sort_values(['일자'], ascending=[True])
df2=df2.sort_values(['일자'], ascending=[True])

# closing_price_samsung=df1['종가'].unique()
# opening_price_samsung=df1['종가'].unique()
Trading_volume_samsung=df1['거래량'].unique()

# closing_price_kioom=df2['종가'].unique()
# opening_price_kioom=df2['종가'].unique()
Trading_volume_kioom=df2['거래량'].unique()

# print(type(closing_price_samsung))
# print(type(closing_price_kioom))

x1=df1.drop(range(893,1120), axis=0)
# print(df1.info) 
# print(df1.shape) #(893, 17)

x2=df2.drop(range(893,1060), axis=0)
# print(df2.info)
# print(df2.shape) #(893, 17)

# df1=df1.loc[::-1].reset_index(drop=True).head(10)
# df2=df2.loc[::-1].reset_index(drop=True).head(10)

#print(df1.describe)
# print(df2.describe)

# print(df1.columns)
# print(df2.columns)

x1=x1.loc[::-1].reset_index(drop=True)
x2=x2.loc[::-1].reset_index(drop=True)

x1=x1.drop(columns=['일자','시가','고가','저가','종가', "Unnamed: 6", '전일비' , '등락률','신용비','개인','기관', '외인(수량)','외국계','프로그램','외인비'], axis=1)
x2=x2.drop(columns=['일자','시가','고가','저가','종가', "Unnamed: 6", '전일비' , '등락률','신용비','개인','기관', '외인(수량)','외국계','프로그램','외인비'], axis=1)
# print(x1.describe) #거래량  금액(백만)   
# print(x2.describe) #거래량  금액(백만)   

x1=np.array(x1)
x2=np.array(x2)
#print(x1.shape, x2.shape)  #(893, 2) (893, 2)

y1=df1['거래량']
y2=df2['거래량']

#print(y1.shape) #(1120,)
#print(y2.shape) #(1060,)

#print(x1.shape,x2.shape,y1.shape,y2.shape) #(893, 2) (893, 2) (1120,) (1060,)

def split_xy3(dataset, time_steps, y_column):
    x,y=list(), list()
    for i in range(len(dataset)):
        x_end_number=i+time_steps
        y_end_number=x_end_number+y_column-1
        
        if y_end_number>len(dataset):
            break
        tmp_x=dataset[i:x_end_number, 1:]
        tmp_y=dataset[x_end_number-1:y_end_number,0]
        x.append(tmp_x)
        y.append(tmp_y)
    return np.array(x), np.array(y)

x1,y1=split_xy3(x1,5,3)
x2,y2=split_xy3(x2,5,3)

# y1=np.log1p(y1)
# y2=np.log1p(y2)


# print(x1.shape, y1.shape) #(887, 5, 1) (887, 3)
# print(x2.shape, y2.shape)  #(887, 5, 1) (887, 3)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler

from sklearn.model_selection import train_test_split


x1_train, x1_test, x2_train, x2_test, y1_train, y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2, train_size=0.8, random_state=66)

#print(x1_train.shape) (709, 5, 1)

# scaler=MinMaxScaler()

# scaler.fit(x1_train)
# x1=scaler.transform(x1)
# x2=scaler.transform(x2)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(5,1))
dense1=LSTM(5, activation='relu', name='dense1')(input1)
dense2=Dense(10, activation='relu', name='dense2')(dense1)
dense3=Dense(8, activation='relu', name='dense3')(dense2)
output1=Dense(5, activation='relu', name='dense4')(dense3)

input2=Input(shape=(5,1))
dense11=LSTM(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(6, activation='relu', name='dense13')(dense12)
dense14=Dense(4, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])

output21 = Dense(8)(merge1)
output22 = Dense(6)(output21)
output23 = Dense(4, activation='relu')(output22)
last_output1 = Dense(1)(output23)

output31 = Dense(10)(merge1)
output32 = Dense(8)(output31)
output33 = Dense(6, activation='relu')(output32)
output34 = Dense(4, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model=Model(inputs=[input1,input2], outputs=[last_output1, last_output2])


model.compile(loss='mse', optimizer='adam', metrics=['mse'])


############################################################################################################
import datetime
date=datetime.datetime.now()   
datetime = date.strftime("%m%d_%H%M")   #1206_0456
#print(datetime)

filepath='./_ModelCheckPoint/'
filename='{epoch:04d}-{val_loss:.4f}.hdf5'  # 2500-0.3724.hdf
model_path = "".join([filepath, 'k26_', datetime, '_', filename])
            #./_ModelCheckPoint/k26_1206_0456_2500-0.3724.hdf
##############################################################################################################

es = EarlyStopping(monitor='val_loss', patience=50, mode='auto', verbose=1, restore_best_weights=True)
mcp=ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, 
                    save_best_only=True,
                    filepath=model_path)

start=time.time()
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=1000, batch_size=16, validation_split=0.2, verbose=1, callbacks=[es, mcp]) 
end=time.time()-start

print("걸린시간 :" , round(end, 3), '초')


# print(df1.shape) #(893, 4)
# print(df1.info) #시가, 고가, 저가, 거래량

# print(df2.columns)

model.save_weights("../study/_save/samsung stock price 거래량 제출.h5")

#mode=load_model("../study/_save/samsung stock price.h5")

loss=model.evaluate([x1_test, x2_test], [y1_test, y2_test], batch_size=1)

y1_pred,y2_pred=model.predict([x1, x2])

# y1_pred1=np.expm1(y1_pred)
# y2_pred2=np.expm1(y2_pred)

# print(y1_pred[-1], y2_pred[-1])

print('삼성예측거래량: ' , y1_pred[-1][-1])
print('키움예측거래량: ' , y2_pred[-1][-1])


        
