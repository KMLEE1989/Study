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

df2=pd.read_csv(path+'미세먼지.csv',thousands=',')

# df2.info()
df2['DATE']=df2['DATE'].astype('datetime64[ns]')

# df2.info()

print(df2.sort_values(by='미세먼지 PM2.5').head())
print(df2.sort_values(by='미세먼지 PM10').head())


font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=18)

print(df2['DATE'])
print(df2['미세먼지 PM2.5'])
print(df2['미세먼지 PM10'])

plt.figure(figsize=(100,100))
plt.plot(df2['DATE'], df2['미세먼지 PM2.5'], df2['DATE'], df2['미세먼지 PM10'], 'rs-', marker='o')
plt.title('날짜별 미세먼지', font=fontprop)
plt.ylabel('미세먼지 PM2.5, 미세먼지 PM10', font=fontprop)
plt.xlabel('DATE', font=fontprop)
plt.legend(['PM2.5','PM10'], loc='upper right', ncol=1, fontsize=10)
plt.grid()
plt.show()


"""
df2_dat.plot()
# print(max(x1['미세먼지 PM2.5'])) 74 시각화를 위한 준비
# print(min(x1['미세먼지 PM2.5'])) 4   시각화를 위한 준비

# print(max(x1['미세먼지 PM10']))  414  시각화를 위한 준비
# print(min(x1['미세먼지 PM10']))  8  시각화를 위한 준비
"""


#x1=df2


"""
df2_dat.plot()
# print(max(x1['미세먼지 PM2.5'])) 74 시각화를 위한 준비
# print(min(x1['미세먼지 PM2.5'])) 4   시각화를 위한 준비

# print(max(x1['미세먼지 PM10']))  414  시각화를 위한 준비
# print(min(x1['미세먼지 PM10']))  8  시각화를 위한 준비
"""


"""
plt.subplot()
plt.xlabel()/plt.ylabel()
plt.slim()/plt.ylim()
plt.legend()
plt.title()
plt.show()
"""
'''
# x1=x1.drop(['DATE'],axis=1)
# print(x1.describe)
#print(x1.columns)
#print(x1.info())
#x1['미세먼지 PM10'].astype(np.int64)
#x1['미세먼지 PM2.5'].astype(np.int64)
# print(x1.info())
#  #   Column      Non-Null Count  Dtype    
# ---  ------      --------------  -----    
#  0   미세먼지 PM10   712 non-null    int64
#  1   미세먼지 PM2.5  712 non-null    int64
'''


'''
df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')
y1=df1
y1=y1.drop(['DATE', '지점'], axis=1)
# print(y1.columns)
# print(y1.info())

#  #   Column   Non-Null Count  Dtype
# ---  ------   --------------  -----
#  0   평균기온(℃)  712 non-null    float64
#  1   최저기온(℃)  712 non-null    float64
#  2   최고기온(℃)  712 non-null    float64


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

# x1,y1=split_xy3(x,"\n",y)
'''
