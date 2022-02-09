import numpy as np
import pandas as pd
from tensorflow.python.keras.callbacks import EarlyStopping


path = "../_data/SamsungK/"

dataset1=pd.read_csv(path+'삼성전자.csv',thousands=',')

dataset2=pd.read_csv(path+'키움증권.csv',thousands=',')

# print(dataset1)
# print(dataset2)

# print(dataset1.info) #[1120 rows x 17 columns]>
# print(dataset2.info) #[1060 rows x 17 columns]>

# print(dataset1.shape) #(1120, 17)
# print(dataset2.shape) #(1060, 17)

dataset1=dataset1.drop(range(893,1120), axis=0)
# print(dataset1.info) #[893 rows x 17 columns]
# print(dataset1.shape) #(893, 17)
dataset2=dataset2.drop(range(893,1060), axis=0)
# print(dataset2.info) #[893 rows x 17 columns]
# print(dataset2.shape) #(893, 17)

#print(dataset1.columns)
dataset1=dataset1.drop(['일자','종가',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1)
#y1=dataset1['종가']
# print(x1.shape) #(893, 4)
# print(x1.info) #시가, 고가, 저가, 거래량

x2=dataset2.drop(['일자','종가',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1)
y2=dataset2['종가']
# print(x2.shape)  #시가, 고가, 저가, 거래량
# print(x2.info) #(893,4)

a = dataset1
b = a.data
c = a.target

size = 6

def split_x(data, size):
    aaa = []
    for i in range(len(data) - size +1): # 1~10 - 5 +1 //size가 나오는 가장 마지막 것을 생각해서 +1을 해주어야 함
        subset = data[i : (i+size)]      # [1 : 6] = 1,2,3,4,5 
        aaa.append(subset)                  # 1,2,3,4,5에 []를 붙인다.
    return np.array(aaa)

x = split_x(b,size)
y = split_x(c,size)
print(x.shape)   
print(y.shape)  
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, random_state=42
)
print(x_train.shape) 
print(y_test.shape)
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,random_state=1,test_size=0.3)
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,random_state=2,test_size=0.3)

# print(x1_train.shape) #(625, 4)
# print(x2_train.shape) #(625, 4)
# print(x1_test.shape)  #(268, 4)
# print(x2_test.shape)  #(268, 4)
# print(y1_train.shape)  #(625,)
# print(y2_train.shape)  #(625,)
# print(y1_test.shape)   #(268,)
# print(y2_test.shape)   #(268,)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(4,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

input2=Input(shape=(4,))
dense11=Dense(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])

output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21, activation='relu')(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)

model = Model(inputs=[input1, input2], outputs=[last_output1, last_output2])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
from keras.callbacks import EarlyStopping
early_stopping=EarlyStopping(patience=20)
model.fit([x1_train, x2_train],[y1_train, y2_train], epochs=100, batch_size=8, validation_data=([x1, x2],[y1, y2]), callbacks=[early_stopping])
results=model.evaluate([x1_test, x2_test], [y1_test, y2_test])




