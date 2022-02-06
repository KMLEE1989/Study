import matplotlib
import pandas as pd
import numpy as np
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint 
import time
from tensorflow.keras.layers import Dropout, Dense, SimpleRNN,LSTM
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import matplotlib_inline

path = "../_data/개인프로젝트/CSV/"

df1=pd.read_csv(path+'미세먼지.csv',thousands=',')

df1['DATE']=df1['DATE'].astype('datetime64[ns]')

print(df1.info())

#  #   Column      Non-Null Count  Dtype
# ---  ------      --------------  -----
#  0   DATE        712 non-null    datetime64[ns]
#  1   미세먼지 PM10   712 non-null    int64
#  2   미세먼지 PM2.5  712 non-null    int64

font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=18)

plt.figure(figsize=(1000,1000))
plt.plot(df1['DATE'], df1['미세먼지 PM10'], df1['DATE'], df1['미세먼지 PM2.5'],'rs-', marker='o')
plt.title('날짜별 미세먼지', font=fontprop)
plt.ylabel('PM10, PM2.5', font=fontprop)
plt.xlabel('DATE', font=fontprop)
plt.legend(['PM10','PM2.5'], loc='upper right', ncol=1, fontsize=10)
plt.grid()
plt.show()

"""
df1=df1.drop(['지점'], axis=1)
print(df1.info())

# Data columns (total 4 columns):
#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   DATE     712 non-null    object
#  1   평균기온(℃)  712 non-null    float64
#  2   최저기온(℃)  712 non-null    float64
#  3   최고기온(℃)  712 non-null    float64

df1['DATE']=df1['DATE'].astype('datetime64[ns]')

font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=18)

# print(df1.sort_values(by='평균기온(℃)').head())
# print(df1.sort_values(by='최저기온(℃)').head())
# print(df1.sort_values(by='최고기온(℃)').head())
'''
plt.figure(figsize=(1000,1000))
plt.plot(df1['DATE'], df1['평균기온(℃)'], df1['DATE'], df1['최저기온(℃)'], df1['DATE'],df1['최고기온(℃)'],'rs-', marker='o')
plt.title('날짜별 기온', font=fontprop)
plt.ylabel('기온', font=fontprop)
plt.xlabel('DATE', font=fontprop)
plt.legend(['average','MAX','MIN'], loc='upper right', ncol=1, fontsize=10)
plt.grid()
plt.show()
'''
plt.figure(figsize=(1000,1000))
plt.plot(df1['DATE'], df1['평균기온(℃)'],'rs-',marker='o')
plt.title('날짜별 기온', font=fontprop)
plt.ylabel('평균기온', font=fontprop)
plt.xlabel('DATE', font=fontprop)
plt.legend(['average'], loc='upper right', ncol=1, fontsize=10)
plt.grid()
plt.show()
"""
