import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM

path = "../_data/개인프로젝트/CSV/"

df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')

df2=pd.read_csv(path+'미세먼지.csv',thousands=',')

df4=pd.read_csv(path+'확진자.csv',thousands=',')

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





































