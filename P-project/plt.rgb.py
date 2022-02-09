from turtle import color
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


plt.figure(figsize=(12, 9))
plt.plot([13.809026,1,15.202441,2,9.921253,3,7.1764245,4,21.925842,5,20.747742,6, 9.020304,7,-3.9170082,8,17.997305,9,18.123478,10,23.26715,11,14.718391,12,
21.97149,13,19.250015,14,-0.9076167,15,11.950726,16,22.23779,17,12.599094,18, 12.55218,19,11.613571,20,16.92524,21,22.501865,22,0.5150079,23,23.884912,24,
5.920573,25,10.454158,26,23.070553,27,0.98708266,28,18.051989,29,24.509026,30],marker='o',color='red')
plt.title('Conculsion XGB VER')
plt.xlabel('DATE from January 1')
plt.ylabel('AVG Temp')
plt.legend(['S.korea AVG TEMP per day in january'], loc='upper right', fontsize=10)
plt.show()

# plt.figure(figsize=(12, 9))
# plt.plot(y1_predict[:30], marker='o')
# plt.xlabel('DATE from January 1')
# plt.ylabel('AVG Temp')
# plt.legend(['S.korea AVG TEMP per day in january'], loc='upper right', fontsize=10)
# plt.show()


# [13.809026   15.202441    9.921253    7.1764245  21.925842   20.747742
#   9.020304   -3.9170082  17.997305   18.123478   23.26715    14.718391
#  21.97149    19.250015   -0.9076167  11.950726   22.23779    12.599094
#  12.55218    11.613571   16.92524    22.501865    0.5150079  23.884912
#   5.920573   10.454158   23.070553    0.98708266 18.051989   24.509026  ]