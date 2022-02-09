import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.random.set_seed(777) #하이퍼파라미터 튜닝을 위해 실행시 마다 변수가 같은 초기값 가지게 하기
import numpy as np
import pickle
#matplotlib 패키지 한글 깨짐 처리 시작
import matplotlib.pyplot as plt
import platform
if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
        #!wget "https://www.wfonts.com/download/data/2016/06/13/malgun-gothic/malgun.ttf"
        #!mv malgun.ttf /usr/share/fonts/truetype/
        #import matplotlib.font_manager as fm 
        #fm._rebuild() 
        plt.rc('font', family='Malgun Gothic') 
plt.rcParams['axes.unicode_minus'] = False #한글 폰트 사용시 마이너스 폰트 깨짐 해결
#matplotlib 패키지 한글 깨짐 처리 끝
# matplotlib inline
import os

def load_time_series_data(data, sequence_length):
    #print(data.shape) #(1225, 1)
    #print(sequence_length) #3
    window_length = sequence_length + 1
    x_data = []
    y_data = []
    for i in range(0, len(data) - window_length + 1): #0 1 2 3 4 5 6 7 8 9 | 10
        window = data[i:i + window_length, :]
        x_data.append(window[:-1, :])
        y_data.append(window[-1, [-1]])
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    #print(x_data.shape) #(1222, 3, 1)
    #print(y_data.shape) #(1222, 1)

    return x_data, y_data

##########데이터 로드

df = pd.read_csv('https://raw.githubusercontent.com/kairess/stock_crypto_price_prediction/master/dataset/005930.KS_5y.csv')

##########데이터 분석

print(df.head())
'''
         Date     Open     High      Low    Close  Adj Close    Volume
0  2013-10-30  29700.0  30000.0  29680.0  30000.0  41.274914  10588400
1  2013-10-31  29960.0  30040.0  29300.0  29300.0  40.311840  12647050
2  2013-11-01  29800.0  30000.0  29360.0  30000.0  41.274914  11357700
3  2013-11-04  29840.0  30040.0  29780.0  29980.0  41.247398  10887800
4  2013-11-05  30040.0  30040.0  29440.0  29700.0  40.862167   8009300
'''

print(df.info())
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1225 entries, 0 to 1224
Data columns (total 7 columns):
 #   Column     Non-Null Count  Dtype  
---  ------     --------------  -----  
 0   Date       1225 non-null   object 
 1   Open       1225 non-null   float64
 2   High       1225 non-null   float64
 3   Low        1225 non-null   float64
 4   Close      1225 non-null   float64
 5   Adj Close  1225 non-null   float64
 6   Volume     1225 non-null   int64  
dtypes: float64(5), int64(1), object(1)
memory usage: 67.1+ KB
None
'''

print(df.describe())
'''
               Open          High           Low         Close     Adj Close  \
count   1225.000000   1225.000000   1225.000000   1225.000000   1225.000000   
mean   34234.089796  34563.338776  33899.338776  34228.236735   9980.833215   
std    10344.907378  10440.589222  10232.254253  10332.317328  15687.605829   
min    21360.000000  21480.000000  20660.000000  21340.000000     38.248085   
25%    25880.000000  26020.000000  25620.000000  25840.000000    231.743439   
50%    28980.000000  29160.000000  28620.000000  28960.000000   1252.430420   
75%    45600.000000  46060.000000  45160.000000  45550.000000  14222.215820   
max    57500.000000  57520.000000  56760.000000  57220.000000  52608.718750   

             Volume  
count  1.225000e+03  
mean   1.205036e+07  
std    5.529522e+06  
min    0.000000e+00  
25%    8.774950e+06  
50%    1.077040e+07  
75%    1.371510e+07  
max    6.468130e+07  
'''

##########데이터 전처리

data = df[['Close']].to_numpy()
print(data.shape) #(1225, 1)

transformer = MinMaxScaler()
data = transformer.fit_transform(data)

sequence_length = 3
x_data, y_data = load_time_series_data(data, sequence_length)
print(x_data.shape) #(1222, 3, 1)
print(y_data.shape) #(1222, 1)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, shuffle=False) #시각화를 위해 shuffle=False 옵션 사용
print(x_train.shape) #(855, 3, 1)
print(y_train.shape) #(855, 1)
print(x_test.shape) #(367, 3, 1)
print(y_test.shape) #(367, 1)

if not os.path.exists('models/samsung_electronics_stock_close_price_time_series_regression_model'):
    os.makedirs('models/samsung_electronics_stock_close_price_time_series_regression_model')

with open('models/samsung_electronics_stock_close_price_time_series_regression_model/transformer.pkl', 'wb') as f:
    pickle.dump(transformer, f)

##########모델 생성

input = tf.keras.layers.Input(shape=(sequence_length, 1))
net = tf.keras.layers.LSTM(units=32, activation='relu')(input) 
net = tf.keras.layers.Dense(units=32, activation='relu')(net)
net = tf.keras.layers.Dense(units=1)(net)
model = tf.keras.models.Model(input, net)

##########모델 학습

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test), callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath='models/samsung_electronics_stock_close_price_time_series_regression_model/saved_model', save_best_only=True, verbose=1)]) 

##########모델 검증

##########모델 예측                           

def plot(data, y_predict_train, y_predict_test):
    plt.plot(transformer.inverse_transform(data)[:, [-1]].flatten(), label='실제 종가')

    y_predict_train = transformer.inverse_transform(y_predict_train)
    y_predict_train_plot = np.empty_like(data[:, [0]])
    y_predict_train_plot[:, :] = np.nan
    y_predict_train_plot[sequence_length:len(y_predict_train) + sequence_length, :] = y_predict_train
    plt.plot(y_predict_train_plot.flatten(), label='학습 데이터 예측 종가')

    y_predict_test = transformer.inverse_transform(y_predict_test)
    y_predict_test_plot = np.empty_like(data[:, [0]])
    y_predict_test_plot[:, :] = np.nan
    y_predict_test_plot[len(y_predict_train) + sequence_length:, :] = y_predict_test
    plt.plot(y_predict_test_plot.flatten(), label='테스트 데이터 예측 종가')

    plt.legend()
    plt.show()

y_predict_train = model.predict(x_train)
y_predict_test = model.predict(x_test)
plot(data, y_predict_train, y_predict_test)

x_test = np.array([
        [[30000], [29300], [30000]]
])
x_test = x_test.reshape(-1, 1)
x_test = transformer.transform(x_test)
x_test = x_test.reshape(1, sequence_length, 1)

y_predict = model.predict(x_test)

y_predict = transformer.inverse_transform(y_predict)
print(y_predict[0][0]) #30079.232