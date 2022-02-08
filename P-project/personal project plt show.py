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

list =list([13.809026,   15.202441,    9.921253,    7.1764245,  21.925842,   20.747742,
  9.020304,   -3.9170082,  17.997305,   18.123478,   23.26715,    14.718391,
 21.97149,    19.250015,   -0.9076167,  11.950726,   22.23779,    12.599094,
 12.55218,    11.613571,   16.92524,    22.501865,    0.5150079,  23.884912,
  5.920573,   10.454158,   23.070553,    0.98708266, 18.051989,   24.509026])

#print(list.info())

path = "../_data/개인프로젝트/CSV/"

dft=pd.read_csv(path+'통합.csv',thousands=',')

plt.figure(figsize=(100,100))
plt.plot(list,marker='o', color='black')
plt.title('Conclusion xgbost VER')
plt.ylabel('AVG TEMP')
plt.xlabel('DATE from january1')
plt.legend(['S.Korea AVG TEMP per day in January1'], loc='upper right', ncol=1, fontsize=10)
plt.grid()
plt.show()
