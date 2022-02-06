import numpy as np
from numpy.core.fromnumeric import shape
from numpy.lib.utils import info
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
from tensorflow.python.framework.dtypes import as_dtype
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers import concatenate

path = "../_data/개인프로젝트/CSV/"

df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')
y1=df1
y1=y1.drop(['DATE', '지점'], axis=1)
# print(y1.columns)
# print(y1.info())
y1=np.array(y1)
              
path1="../_data/개인프로젝트/CSV/오염대기/"

df3_1=pd.read_csv(path1+'아황산가스.csv', encoding='cp949')
print(df3_1.columns)
print(df3_1.info())
df3_1=df3_1.drop(['날짜'], axis=1)
xx1=np.array(df3_1)

df3_2=pd.read_csv(path1+'오존.csv', encoding='cp949')
print(df3_2.columns)
print(df3_2.info())
df3_2=df3_2.drop(['날짜'], axis=1)
xx2=np.array(df3_2)

df3_3=pd.read_csv(path1+'이산화질소.csv', encoding='cp949')
print(df3_3.columns)
print(df3_3.info())
df3_3=df3_3.drop(['날짜'], axis=1)
xx3=np.array(df3_3)

df3_4=pd.read_csv(path1+'일산화탄소.csv', encoding='cp949')
print(df3_4.columns)
print(df3_4.info())
df3_4=df3_4.drop(['날짜'], axis=1)
xx4=np.array(df3_4)

xx1_train, xx1_test, xx2_train, xx2_test, xx3_train, xx3_test, xx4_train, xx4_test, y1_train, y1_test = train_test_split(xx1,xx2,xx3,xx4,y1,train_size=0.7, shuffle=True, random_state=66)

# print(xx1_train.shape, xx1_test.shape)  #(498, 17) (214, 17)
# print(xx2_train.shape, xx2_test.shape) #(498, 17) (214, 17)
# print(xx3_train.shape, xx3_test.shape) #(498, 17) (214, 17)
# print(xx4_train.shape, xx4_test.shape) #(498, 17) (214, 17)
# print(y1_train.shape, y1_test.shape) #(498, 3) (214, 3)


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

input1=Input(shape=(17,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

input2=Input(shape=(17,))
dense11=Dense(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

input3=Input(shape=(17,))
dense21=Dense(5, activation='relu', name='dense21')(input3)
dense22=Dense(10, activation='relu', name='dense22')(dense21)
dense23=Dense(7, activation='relu', name='dense23')(dense22)
dense24=Dense(7, activation='relu', name='dense24')(dense23)
output3=Dense(5, activation='relu', name='output3')(dense24)

input4=Input(shape=(17,))
dense31=Dense(5, activation='relu', name='dense31')(input4)
dense32=Dense(10, activation='relu', name='dense32')(dense31)
dense33=Dense(7, activation='relu', name='dense33')(dense32)
dense34=Dense(7, activation='relu', name='dense34')(dense33)
output4=Dense(5, activation='relu', name='output4')(dense34)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2, output3, output4])

output21 = Dense(3)(merge1)
output22 = Dense(10)(output21)
output23 = Dense(10, activation='relu')(output22)
last_output1 = Dense(1)(output23)

model = Model(inputs=[input1, input2, input3,input4], outputs=[last_output1])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([xx1_train, xx2_train, xx3_train,xx4_train], [y1_train], epochs=100, batch_size=32, validation_data=([xx1,xx2,xx3,xx4],[y1]), verbose=1)

result=model.evaluate([xx1_test, xx2_test, xx3_test, xx4_test],[y1_test])

print(result.shape)
