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
print(y1.info())

path1="../_data/개인프로젝트/CSV/오염대기/"

df3_1=pd.read_csv(path1+'아황산가스.csv', encoding='cp949')
print(df3_1.columns)
print(df3_1.info())
df3_1=df3_1.drop(['날짜'], axis=1)

df3_1=np.array(df3_1)
y1=np.array(y1)

# print(df3_1.shape) (712, 17)
# print(y1.shape) (712, 3)

df3_1_train, df3_1_test, y1_train, y1_test = train_test_split(df3_1, y1, train_size=0.7, shuffle=True, random_state=66)

# print(df3_1_train.shape, df3_1_test.shape) (498, 17) (214, 17)
# print(y1_train.shape, y1_test.shape) (498, 3) (214, 3)

input1=Input(shape=(17,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

merge1 = concatenate([output1])
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

"""
df3_2=pd.read_csv(path1+'오존.csv', encoding='cp949')
df3_2=df3_2.drop(['날씨'], axis=1)

df3_3=pd.read_csv(path1+'이산화질소.csv', encoding='cp949')
df3_3=df3_3.drop(['날씨'],axis=1)

df3_4=pd.read_csv(path1+'일산화탄소.csv', encoding='cp949')
df3_4=df3_4.drop(['날씨'],axis=1)
"""

'''
x1=np.array([range(100), range(301,401)])   #삼성 저가 , 고가
x2=np.array([range(101,201), range(411, 511), range(100,200)]) #미국선물 시가, 고가, 종가


x1=np.transpose(x1)
x2=np.transpose(x2)

y1=np.array(range(1001, 1101))    #삼성전자 종가
y2=np.array(range(101,201))

print(x1.shape, x2.shape, y1.shape, y2.shape) #(100, 2) (100, 3) (100,)

from sklearn.model_selection import train_test_split

# x1=np.array(range(100))  
# x2=np.array(range(1,101))

x1_train, x1_test,x2_train,x2_test,y1_train,y1_test, y2_train, y2_test = train_test_split(x1,x2,y1,y2,train_size=0.7, shuffle=True, random_state=66)
# x1_test, x1_val, y_test, y_val=train_test_split(x1_test,y_test,train_size=0.7, shuffle=True, random_state=66)
# x2_test, x2_val, y_test, y_val=train_test_split(x2_test, y_test,train_size=0.7, shuffle=True, random_state=66)

print(x1_train.shape, x1_test.shape)  #(70, 2) (30, 2)
print(x2_test.shape, x2_test.shape) #(30, 3) (30, 3)
print(y1_train.shape, y1_test.shape) #(70,) (30,)
print(y2_train.shape, y2_test.shape) #(#70,) (30,)

#2. 모델구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1 모델1
input1=Input(shape=(2,))
dense1=Dense(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(7, activation='relu', name='dense4')(dense3)

#2-1 모델1
input2=Input(shape=(3,))
dense11=Dense(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

from tensorflow.keras.layers import concatenate, Concatenate

merge1 = concatenate([output1, output2])

#2-3 output 모델1
output21 = Dense(7)(merge1)
output22 = Dense(11)(output21)
output23 = Dense(11, activation='relu')(output22)
last_output1 = Dense(1)(output23)

#2-3 output 모델1
output31 = Dense(7)(merge1)
output32 = Dense(21)(output31)
output33 = Dense(21, activation='relu')(output32)
output34 = Dense(11, activation='relu')(output33)
last_output2 = Dense(1)(output34)



'''













path = "../_data/개인프로젝트/CSV/"

'''
df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')

df2=pd.read_csv(path+'미세먼지.csv',thousands=',')
    
#print(df2.columns) Index(['DATE', '미세먼지 PM10', '미세먼지 PM2.5'], dtype='object')
# df2.at[215, '미세먼지 PM10']='24'
# df2.at[217, '미세먼지 PM10']='24'
# df2.at[218, '미세먼지 PM10']='24'
# df2.at[220, '미세먼지 PM10']='24'
# df2.at[221, '미세먼지 PM10']='24'
# df2.at[222, '미세먼지 PM10']='24'
# df2.at[223, '미세먼지 PM10']='24'
# df2.at[225, '미세먼지 PM10']='24'
# df2.at[226, '미세먼지 PM10']='24'
# df2.at[227, '미세먼지 PM10']='24'
# df2.at[228, '미세먼지 PM10']='24'
# df2.at[231, '미세먼지 PM10']='22'
# df2.at[232, '미세먼지 PM10']='22'
# df2.at[234, '미세먼지 PM10']='22'
# df2.at[240, '미세먼지 PM10']='22'
# df2.at[273, '미세먼지 PM10']='22'
# df2.at[278, '미세먼지 PM10']='34'
# df2.at[279, '미세먼지 PM10']='34'
# df2.at[280, '미세먼지 PM10']='34'
# df2.at[281, '미세먼지 PM10']='34'
# df2.at[283, '미세먼지 PM10']='34'
# df2.at[284, '미세먼지 PM10']='34'
# df2.at[285, '미세먼지 PM10']='34'
# df2.at[286, '미세먼지 PM10']='34'
# df2.at[288, '미세먼지 PM10']='38'
# df2.at[289, '미세먼지 PM10']='38'
# df2.at[291, '미세먼지 PM10']='38'
# df2.at[292, '미세먼지 PM10']='38'
# df2.at[294, '미세먼지 PM10']='38'
# df2.at[295, '미세먼지 PM10']='38'
# df2.at[297, '미세먼지 PM10']='38'
# df2.at[301, '미세먼지 PM10']='38'
# df2.at[419, '미세먼지 PM10']='65'
# df2.at[420, '미세먼지 PM10']='65'
# df2.at[461, '미세먼지 PM10']='41'
# df2.at[530, '미세먼지 PM10']='33'

# df2.at[215, '미세먼지 PM2.5']='14'
# df2.at[217, '미세먼지 PM2.5']='14'
# df2.at[218, '미세먼지 PM2.5']='14'
# df2.at[220, '미세먼지 PM2.5']='14'
# df2.at[221, '미세먼지 PM2.5']='14'
# df2.at[222, '미세먼지 PM2.5']='14'
# df2.at[223, '미세먼지 PM2.5']='14'
# df2.at[225, '미세먼지 PM2.5']='14'
# df2.at[226, '미세먼지 PM2.5']='14'
# df2.at[227, '미세먼지 PM2.5']='14'
# df2.at[228, '미세먼지 PM2.5']='14'
# df2.at[231, '미세먼지 PM2.5']='12'
# df2.at[232, '미세먼지 PM2.5']='12'
# df2.at[234, '미세먼지 PM2.5']='12'
# df2.at[240, '미세먼지 PM2.5']='12'
# df2.at[273, '미세먼지 PM2.5']='17'
# df2.at[278, '미세먼지 PM2.5']='17'
# df2.at[279, '미세먼지 PM2.5']='17'
# df2.at[280, '미세먼지 PM2.5']='17'
# df2.at[281, '미세먼지 PM2.5']='17'
# df2.at[283, '미세먼지 PM2.5']='17'
# df2.at[284, '미세먼지 PM2.5']='17'
# df2.at[285, '미세먼지 PM2.5']='17'
# df2.at[286, '미세먼지 PM2.5']='17'
# df2.at[288, '미세먼지 PM2.5']='21'
# df2.at[289, '미세먼지 PM2.5']='21'
# df2.at[291, '미세먼지 PM2.5']='21'
# df2.at[292, '미세먼지 PM2.5']='21'
# df2.at[294, '미세먼지 PM2.5']='21'
# df2.at[295, '미세먼지 PM2.5']='21'
# df2.at[297, '미세먼지 PM2.5']='21'
# df2.at[301, '미세먼지 PM2.5']='21'
# df2.at[419, '미세먼지 PM2.5']='29'
# df2.at[420, '미세먼지 PM2.5']='26'
# df2.at[461, '미세먼지 PM2.5']='14'
# df2.at[530, '미세먼지 PM2.5']='21'

df4=pd.read_csv(path+'확진자.csv',thousands=',')
df4=df4.replace("-","")

path1="../_data/개인프로젝트/CSV/오염대기/"

df3_1=pd.read_csv(path1+'아황산가스.csv', encoding='cp949')
df3_2=pd.read_csv(path1+'오존.csv', encoding='cp949')
df3_3=pd.read_csv(path1+'이산화질소.csv', encoding='cp949')
df3_4=pd.read_csv(path1+'일산화탄소.csv', encoding='cp949')

import csv
import glob
import os

"""
input_path="../_data/개인프로젝트/CSV/오염대기/"
output_file="../_data/개인프로젝트/CSV/오염대기/merge.csv"


file_list=glob.glob(input_path + '*.csv')


with open(output_file, 'w') as f:
    for i, file in enumerate (file_list):
        if i==0:
            with open(file, 'r') as f2:
                while True:
                    line=f2.readline()
                    
                    if not line:
                        break
                    
                    f.write(line)
                    
            file_name=file.split('\\')[-1]
            print(file.split('\\')[-1] +  'write complete...')
            
        else:
            with open(file, 'r') as f2:
                n=0
                while True:
                    line=f2.readline()
                    if n!=0:
                        f.write(line)
                        
                    if not line:
                        break
                    n+=1
            
            file_name=file.split('\\')[-1]
            print(file.split('\\')[-1] + 'wrte complete....')
            
#print('>>> All file merge complete....')  

"""      

path2="../_data/개인프로젝트/CSV/오염대기/combine/"

df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')

df2=pd.read_csv(path+'미세먼지.csv',thousands=',')

df3=pd.read_csv(path2+'combine air pollution.csv', thousands=',')

df4=pd.read_csv(path+'확진자.csv',thousands=',')

y1=df1
y1=y1.drop(['DATE', '지점'], axis=1)
# print(y1.columns)
#print(y1.info())

x1=df2
x1=x1.drop(['DATE'],axis=1)
#print(x1.columns)
#print(x1.info())
#x1['미세먼지 PM10'].astype(np.int64)
#x1['미세먼지 PM2.5'].astype(np.int64)
#print(x1.info())

x2=df3
x2=x2.drop(['날짜'],axis=1)
#print(x2.columns)
#print(x2.info())

x3=df4
x3=x3.drop(['DATE', 'NUM', 'INSPECTION'],axis=1)
#print(x3.info())

x1=np.array(x1)
x2=np.array(x2)
x3=np.array(x3)
y1=np.array(y1)

#print(x1.shape, x2.shape, x3.shape, y1.shape) #(712, 2) (2848, 17) (712, 4) (712, 3)

x1=x1.reshape(712,2,1)
x2=x2.reshape(712,68,1)
x3=x3.reshape(712,4,1)
y1=y1.reshape(712,3,1)

print(x1.shape, x2.shape, x3.shape, y1.shape) #(712, 2, 1) (712, 68, 1) (712, 4, 1) (712, 3, 1)

# print(type(x1))
# print(type(x2))
# print(type(x3))
# print(type(y1))

from sklearn.model_selection import train_test_split
x1_train, x1_test, x2_train, x2_test, x3_train, x3_test, y1_train, y1_test = train_test_split(x1,x2,x3,y1, train_size=0.7, shuffle=True, random_state=66)

# print(x1_train.shape, x1_test.shape) (498, 2, 1) (214, 2, 1)
# print(x2_train.shape, x2_test.shape) (498, 68, 1) (214, 68, 1)
# print(x3_train.shape, x3_test.shape) (498, 4, 1) (214, 4, 1)
# print(y1_train.shape, y1_test.shape) (498, 3, 1) (214, 3, 1)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.layers import concatenate,Concatenate

input1=Input(shape=(2,1))
dense1=LSTM(5, activation='relu', name='dense1')(input1)
dense2=Dense(7, activation='relu', name='dense2')(dense1)
dense3=Dense(7, activation='relu', name='dense3')(dense2)
output1=Dense(5, activation='relu', name='output1')(dense3)

input2=Input(shape=(68,1))
dense11=LSTM(5, activation='relu', name='dense11')(input2)
dense12=Dense(10, activation='relu', name='dense12')(dense11)
dense13=Dense(7, activation='relu', name='dense13')(dense12)
dense14=Dense(7, activation='relu', name='dense14')(dense13)
output2=Dense(5, activation='relu', name='output2')(dense14)

input3=Input(shape=(4,1))
dense21=LSTM(5, activation='relu', name='dense21')(input3)
dense22=Dense(10, activation='relu', name='dense22')(dense21)
dense23=Dense(7, activation='relu', name='dense23')(dense22)
dense24=Dense(7, activation='relu', name='dense24')(dense23)
output3=Dense(5, activation='relu', name='output3')(dense24)

merge1=concatenate([output1,output2,output3])
merge2 = Dense(10, activation='relu')(merge1)
merge3 = Dense(11)(merge2)
merge4 = Dense(11, activation='relu')(merge3)
merge5 = Dense(5,activation='relu')(merge4)
last_output1 = Dense(1)(merge5)

model = Model(inputs=[input1, input2, input3], outputs=[last_output1])

model.compile(loss='mse', optimizer='adam', metrics=['mse'])

model.fit([x1_train, x2_train, x3_train], [y1_train], epochs=1, batch_size=8, validation_data=([x1,x2,x3],[y1]), verbose=1)

loss=model.evaluate([x1_test,x2_test,x3_test],[y1_test])

y1_predict= model.predict([x1_test, x2_test, x3_test])

print("대한민국의 평균 기온: ", y1_predict)

'''