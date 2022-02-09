import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

%matplotlib inline
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'NanumGothic'

import FinanceDataReader as fdr

# 삼성전자(005930) 전체 (1996-11-05 ~ 현재)
samsung = fdr.DataReader('005930')

# 매우 편리하게 삼성전자 주가 데이터를 DataFrame형식으로 받아옵니다.

# 기본 오름차순 정렬이 된 데이터임을 알 수 있습니다.

# 컬럼 설명
# Open: 시가
# High: 고가
# Low: 저가
# Close: 종가
# Volume: 거래량
# Change: 대비

samsung.tail()


# Open	High	Low	Close	Volume	Change
# Date						
# 2020-12-18	73300	73700	73000	73000	17613029	-0.004093
# 2020-12-21	73100	73400	72000	73000	20367355	0.000000
# 2020-12-22	72500	73200	72100	72300	16304910	-0.009589
# 2020-12-23	72400	74000	72300	73900	19411326	0.022130
# 2020-12-24	74100	78800	74000	77800	32317535	0.052774

# Apple(AAPL), 애플
apple = fdr.DataReader('AAPL')

apple.tail()

# 	Close	Open	High	Low	Volume	Change
# Date						
# 2020-12-18	126.65	128.96	129.10	126.12	192540000.0	-0.0159
# 2020-12-21	128.23	125.03	128.26	123.47	121250000.0	0.0124
# 2020-12-22	131.88	131.68	134.40	129.66	169350000.0	0.0285
# 2020-12-23	130.96	132.18	132.32	130.83	88220000.0	-0.0070
# 2020-12-24	131.99	131.19	133.46	131.10	52790000.0	0.0079

# Apple(AAPL), 애플
apple = fdr.DataReader('AAPL', '2017')
apple.head()

# 	Close	Open	High	Low	Volume	Change
# Date						
# 2017-01-03	29.04	28.95	29.08	28.69	115130000.0	0.0031
# 2017-01-04	29.00	28.96	29.13	28.94	84470000.0	-0.0014
# 2017-01-05	29.15	28.98	29.22	28.95	88770000.0	0.0052
# 2017-01-06	29.48	29.20	29.54	29.12	127010000.0	0.0113
# 2017-01-09	29.75	29.49	29.86	29.48	134250000.0	0.0092

# Ford(F), 1980-01-01 ~ 2019-12-30 (40년 데이터)
ford = fdr.DataReader('F', '1980-01-01', '2019-12-30')
ford.head()

# 	Close	Open	High	Low	Volume	Change
# Date						
# 1980-03-18	1.85	1.85	1.88	1.84	3770000.0	-0.0160
# 1980-03-19	1.87	1.87	1.88	1.85	1560000.0	0.0108
# 1980-03-20	1.88	1.88	1.90	1.87	1450000.0	0.0053
# 1980-03-21	1.80	1.80	1.87	1.78	5020000.0	-0.0426
# 1980-03-24	1.73	1.73	1.77	1.68	3330000.0	-0.0389

ford.tail()

# 	Close	Open	High	Low	Volume	Change
# Date						
# 2019-12-23	9.44	9.50	9.57	9.40	54800000.0	-0.0042
# 2019-12-24	9.47	9.44	9.49	9.43	11880000.0	0.0032
# 2019-12-26	9.45	9.47	9.49	9.43	28980000.0	-0.0021
# 2019-12-27	9.36	9.45	9.46	9.35	28270000.0	-0.0095
# 2019-12-30	9.25	9.34	9.35	9.23	36090000.0	-0.0118


# 삼성전자 주식코드: 005930
STOCK_CODE = '005930'
stock = fdr.DataReader(STOCK_CODE)
stock.head()

# 	Open	High	Low	Close	Volume	Change
# Date						
# 1997-01-20	800	844	800	838	91310	NaN
# 1997-01-21	844	844	803	809	81800	-0.034606
# 1997-01-22	805	805	782	786	81910	-0.028430
# 1997-01-23	786	798	770	776	74200	-0.012723
# 1997-01-24	745	793	745	783	98260	0.009021

stock.tail()
# 	Open	High	Low	Close	Volume	Change
# Date						
# 2020-12-18	73300	73700	73000	73000	17613029	-0.004093
# 2020-12-21	73100	73400	72000	73000	20367355	0.000000
# 2020-12-22	72500	73200	72100	72300	16304910	-0.009589
# 2020-12-23	72400	74000	72300	73900	19411326	0.022130
# 2020-12-24	74100	78800	74000	77800	32317535	0.052774

stock.index
DatetimeIndex(['1997-01-20', '1997-01-21', '1997-01-22', '1997-01-23',
               '1997-01-24', '1997-01-25', '1997-01-27', '1997-01-28',
               '1997-01-29', '1997-01-30',
               ...
               '2020-12-11', '2020-12-14', '2020-12-15', '2020-12-16',
               '2020-12-17', '2020-12-18', '2020-12-21', '2020-12-22',
               '2020-12-23', '2020-12-24'],
              dtype='datetime64[ns]', name='Date', length=6000, freq=None)

stock['Year'] = stock.index.year
stock['Month'] = stock.index.month
stock['Day'] = stock.index.day

stock.head()

# 	Open	High	Low	Close	Volume	Change	Year	Month	Day
# Date									
# 1997-01-20	800	844	800	838	91310	NaN	1997	1	20
# 1997-01-21	844	844	803	809	81800	-0.034606	1997	1	21
# 1997-01-22	805	805	782	786	81910	-0.028430	1997	1	22
# 1997-01-23	786	798	770	776	74200	-0.012723	1997	1	23
# 1997-01-24	745	793	745	783	98260	0.009021	1997	1	24

plt.figure(figsize=(16, 9))
sns.lineplot(y=stock['Close'], x=stock.index)
plt.xlabel('time')
plt.ylabel('price')

time_steps = [['1990', '2000'], 
              ['2000', '2010'], 
              ['2010', '2015'], 
              ['2015', '2020']]

fig, axes = plt.subplots(2, 2)
fig.set_size_inches(16, 9)
for i in range(4):
    ax = axes[i//2, i%2]
    df = stock.loc[(stock.index > time_steps[i][0]) & (stock.index < time_steps[i][1])]
    sns.lineplot(y=df['Close'], x=df.index, ax=ax)
    ax.set_title(f'{time_steps[i][0]}~{time_steps[i][1]}')
    ax.set_xlabel('time')
    ax.set_ylabel('price')
plt.tight_layout()
plt.show()

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# 스케일을 적용할 column을 정의합니다.
scale_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
# 스케일 후 columns
scaled = scaler.fit_transform(stock[scale_cols])
scaled

# array([[0.01079622, 0.01071066, 0.01081081, 0.00273412, 0.00143815],
#        [0.01139001, 0.01071066, 0.01085135, 0.00235834, 0.00128837],
#        [0.0108637 , 0.01021574, 0.01056757, 0.00206031, 0.0012901 ],
#        ...,
#        [0.97840756, 0.92893401, 0.97432432, 0.92873155, 0.25680619],
#        [0.97705803, 0.93908629, 0.97702703, 0.94946419, 0.30573298],
#        [1.        , 1.        , 1.        , 1.        , 0.50900883]])

df = pd.DataFrame(scaled, columns=scale_cols)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(df.drop('Close', 1), df['Close'], test_size=0.2, random_state=0, shuffle=False)
x_train.shape, y_train.shape
((4800, 4), (4800,))
x_test.shape, y_test.shape
((1200, 4), (1200,))

x_train

Open	High	Low	Volume
0	0.010796	0.010711	0.010811	0.001438
1	0.011390	0.010711	0.010851	0.001288
2	0.010864	0.010216	0.010568	0.001290
3	0.010607	0.010127	0.010405	0.001169
4	0.010054	0.010063	0.010068	0.001548
...	...	...	...	...
4795	0.307692	0.291878	0.301622	0.006883
4796	0.310931	0.295178	0.311081	0.004095
4797	0.313360	0.295939	0.310000	0.002620
4798	0.310391	0.292386	0.307297	0.002749
4799	0.310391	0.294670	0.310270	0.003905
4800 rows × 4 columns

import tensorflow as tf

def windowed_dataset(series, window_size, batch_size, shuffle):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.map(lambda w: (w[:-1], w[-1]))
    return ds.batch(batch_size).prefetch(1)

WINDOW_SIZE=20
BATCH_SIZE=32
# trian_data는 학습용 데이터셋, test_data는 검증용 데이터셋 입니다.
train_data = windowed_dataset(y_train, WINDOW_SIZE, BATCH_SIZE, True)
test_data = windowed_dataset(y_test, WINDOW_SIZE, BATCH_SIZE, False)

# 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
    
     ]
# 아래의 코드로 데이터셋의 구성을 확인해 볼 수 있습니다.
# X: (batch_size, window_size, feature)
# Y: (batch_size, feature)
for data in train_data.take(1):
    print(f'데이터셋(X) 구성(batch_size, window_size, feature갯수): {data[0].shape}')
    print(f'데이터셋(Y) 구성(batch_size, window_size, feature갯수): {data[1].shape}')
데이터셋(X) 구성(batch_size, window_size, feature갯수): (32, 20, 1)
데이터셋(Y) 구성(batch_size, window_size, feature갯수): (32, 1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, Lambda
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


model = Sequential([
    # 1차원 feature map 생성
    Conv1D(filters=32, kernel_size=5,
           padding="causal",
           activation="relu",
           input_shape=[WINDOW_SIZE, 1]),
    # LSTM
    LSTM(16, activation='tanh'),
    Dense(16, activation="relu"),
    Dense(1),
])

# Sequence 학습에 비교적 좋은 퍼포먼스를 내는 Huber()를 사용합니다.
loss = Huber()
optimizer = Adam(0.0005)
model.compile(loss=Huber(), optimizer=optimizer, metrics=['mse'])

# earlystopping은 10번 epoch통안 val_loss 개선이 없다면 학습을 멈춥니다.
earlystopping = EarlyStopping(monitor='val_loss', patience=10)
# val_loss 기준 체크포인터도 생성합니다.
filename = os.path.join('tmp', 'ckeckpointer.ckpt')
checkpoint = ModelCheckpoint(filename, 
                             save_weights_only=True, 
                             save_best_only=True, 
                             monitor='val_loss', 
                             verbose=1)

history = model.fit(train_data, 
                    validation_data=(test_data), 
                    epochs=50, 
                    callbacks=[checkpoint, earlystopping])

Epoch 1/50
    145/Unknown - 1s 6ms/step - loss: 1.3915e-04 - mse: 2.7831e-04
Epoch 00001: val_loss improved from inf to 0.00442, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 2s 11ms/step - loss: 1.3702e-04 - mse: 2.7404e-04 - val_loss: 0.0044 - val_mse: 0.0088
Epoch 2/50
150/150 [==============================] - ETA: 0s - loss: 3.5403e-05 - mse: 7.0805e-05
Epoch 00002: val_loss improved from 0.00442 to 0.00373, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 3.5403e-05 - mse: 7.0805e-05 - val_loss: 0.0037 - val_mse: 0.0075
Epoch 3/50
145/150 [============================>.] - ETA: 0s - loss: 3.0890e-05 - mse: 6.1779e-05
Epoch 00003: val_loss improved from 0.00373 to 0.00265, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 3.2327e-05 - mse: 6.4655e-05 - val_loss: 0.0026 - val_mse: 0.0053
Epoch 4/50
139/150 [==========================>...] - ETA: 0s - loss: 3.0849e-05 - mse: 6.1697e-05
Epoch 00004: val_loss did not improve from 0.00265
150/150 [==============================] - 1s 8ms/step - loss: 3.3384e-05 - mse: 6.6768e-05 - val_loss: 0.0028 - val_mse: 0.0056
Epoch 5/50
140/150 [===========================>..] - ETA: 0s - loss: 2.9329e-05 - mse: 5.8658e-05
Epoch 00005: val_loss improved from 0.00265 to 0.00255, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 3.1223e-05 - mse: 6.2445e-05 - val_loss: 0.0026 - val_mse: 0.0051
Epoch 6/50
139/150 [==========================>...] - ETA: 0s - loss: 2.9247e-05 - mse: 5.8493e-05
Epoch 00006: val_loss improved from 0.00255 to 0.00206, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 3.1141e-05 - mse: 6.2282e-05 - val_loss: 0.0021 - val_mse: 0.0041
Epoch 7/50
146/150 [============================>.] - ETA: 0s - loss: 2.9408e-05 - mse: 5.8817e-05
Epoch 00007: val_loss did not improve from 0.00206
150/150 [==============================] - 1s 8ms/step - loss: 2.9931e-05 - mse: 5.9862e-05 - val_loss: 0.0022 - val_mse: 0.0044
Epoch 8/50
139/150 [==========================>...] - ETA: 0s - loss: 2.6324e-05 - mse: 5.2648e-05
Epoch 00008: val_loss did not improve from 0.00206
150/150 [==============================] - 1s 8ms/step - loss: 2.8588e-05 - mse: 5.7175e-05 - val_loss: 0.0022 - val_mse: 0.0043
Epoch 9/50
147/150 [============================>.] - ETA: 0s - loss: 2.6072e-05 - mse: 5.2143e-05
Epoch 00009: val_loss improved from 0.00206 to 0.00140, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.6393e-05 - mse: 5.2786e-05 - val_loss: 0.0014 - val_mse: 0.0028
Epoch 10/50
142/150 [===========================>..] - ETA: 0s - loss: 2.5399e-05 - mse: 5.0798e-05
Epoch 00010: val_loss did not improve from 0.00140
150/150 [==============================] - 1s 8ms/step - loss: 2.6429e-05 - mse: 5.2859e-05 - val_loss: 0.0015 - val_mse: 0.0031
Epoch 11/50
146/150 [============================>.] - ETA: 0s - loss: 2.4973e-05 - mse: 4.9946e-05
Epoch 00011: val_loss did not improve from 0.00140
150/150 [==============================] - 1s 8ms/step - loss: 2.5656e-05 - mse: 5.1313e-05 - val_loss: 0.0019 - val_mse: 0.0038
Epoch 12/50
143/150 [===========================>..] - ETA: 0s - loss: 2.3122e-05 - mse: 4.6245e-05
Epoch 00012: val_loss improved from 0.00140 to 0.00131, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.4026e-05 - mse: 4.8053e-05 - val_loss: 0.0013 - val_mse: 0.0026
Epoch 13/50
145/150 [============================>.] - ETA: 0s - loss: 2.3306e-05 - mse: 4.6611e-05
Epoch 00013: val_loss improved from 0.00131 to 0.00073, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.3637e-05 - mse: 4.7273e-05 - val_loss: 7.2618e-04 - val_mse: 0.0015
Epoch 14/50
146/150 [============================>.] - ETA: 0s - loss: 2.3101e-05 - mse: 4.6202e-05
Epoch 00014: val_loss did not improve from 0.00073
150/150 [==============================] - 1s 8ms/step - loss: 2.3758e-05 - mse: 4.7515e-05 - val_loss: 0.0011 - val_mse: 0.0021
Epoch 15/50
138/150 [==========================>...] - ETA: 0s - loss: 2.0612e-05 - mse: 4.1225e-05
Epoch 00015: val_loss did not improve from 0.00073
150/150 [==============================] - 1s 8ms/step - loss: 2.2132e-05 - mse: 4.4265e-05 - val_loss: 9.7743e-04 - val_mse: 0.0020
Epoch 16/50
145/150 [============================>.] - ETA: 0s - loss: 2.1968e-05 - mse: 4.3937e-05
Epoch 00016: val_loss improved from 0.00073 to 0.00068, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.2618e-05 - mse: 4.5237e-05 - val_loss: 6.7990e-04 - val_mse: 0.0014
Epoch 17/50
140/150 [===========================>..] - ETA: 0s - loss: 2.0500e-05 - mse: 4.1001e-05
Epoch 00017: val_loss improved from 0.00068 to 0.00051, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.2256e-05 - mse: 4.4512e-05 - val_loss: 5.1027e-04 - val_mse: 0.0010
Epoch 18/50
141/150 [===========================>..] - ETA: 0s - loss: 1.9266e-05 - mse: 3.8533e-05
Epoch 00018: val_loss improved from 0.00051 to 0.00047, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 2.0363e-05 - mse: 4.0727e-05 - val_loss: 4.7082e-04 - val_mse: 9.4163e-04
Epoch 19/50
140/150 [===========================>..] - ETA: 0s - loss: 1.8217e-05 - mse: 3.6433e-05
Epoch 00019: val_loss improved from 0.00047 to 0.00039, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.8908e-05 - mse: 3.7815e-05 - val_loss: 3.9196e-04 - val_mse: 7.8393e-04
Epoch 20/50
146/150 [============================>.] - ETA: 0s - loss: 1.7757e-05 - mse: 3.5514e-05
Epoch 00020: val_loss did not improve from 0.00039
150/150 [==============================] - 1s 8ms/step - loss: 1.8188e-05 - mse: 3.6375e-05 - val_loss: 5.7883e-04 - val_mse: 0.0012
Epoch 21/50
143/150 [===========================>..] - ETA: 0s - loss: 1.8277e-05 - mse: 3.6553e-05
Epoch 00021: val_loss improved from 0.00039 to 0.00037, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 9ms/step - loss: 1.9064e-05 - mse: 3.8128e-05 - val_loss: 3.6557e-04 - val_mse: 7.3113e-04
Epoch 22/50
149/150 [============================>.] - ETA: 0s - loss: 1.7973e-05 - mse: 3.5946e-05
Epoch 00022: val_loss did not improve from 0.00037
150/150 [==============================] - 1s 8ms/step - loss: 1.7972e-05 - mse: 3.5943e-05 - val_loss: 7.2105e-04 - val_mse: 0.0014
Epoch 23/50
144/150 [===========================>..] - ETA: 0s - loss: 1.6506e-05 - mse: 3.3012e-05
Epoch 00023: val_loss improved from 0.00037 to 0.00022, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.6973e-05 - mse: 3.3946e-05 - val_loss: 2.2470e-04 - val_mse: 4.4941e-04
Epoch 24/50
143/150 [===========================>..] - ETA: 0s - loss: 1.5503e-05 - mse: 3.1005e-05
Epoch 00024: val_loss improved from 0.00022 to 0.00020, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.6053e-05 - mse: 3.2106e-05 - val_loss: 1.9826e-04 - val_mse: 3.9652e-04
Epoch 25/50
141/150 [===========================>..] - ETA: 0s - loss: 1.4575e-05 - mse: 2.9149e-05
Epoch 00025: val_loss did not improve from 0.00020
150/150 [==============================] - 1s 8ms/step - loss: 1.5453e-05 - mse: 3.0907e-05 - val_loss: 3.3893e-04 - val_mse: 6.7787e-04
Epoch 26/50
141/150 [===========================>..] - ETA: 0s - loss: 1.4477e-05 - mse: 2.8954e-05
Epoch 00026: val_loss did not improve from 0.00020
150/150 [==============================] - 1s 8ms/step - loss: 1.5228e-05 - mse: 3.0457e-05 - val_loss: 3.3818e-04 - val_mse: 6.7637e-04
Epoch 27/50
146/150 [============================>.] - ETA: 0s - loss: 1.5209e-05 - mse: 3.0417e-05
Epoch 00027: val_loss did not improve from 0.00020
150/150 [==============================] - 1s 8ms/step - loss: 1.5293e-05 - mse: 3.0586e-05 - val_loss: 2.2337e-04 - val_mse: 4.4673e-04
Epoch 28/50
141/150 [===========================>..] - ETA: 0s - loss: 1.4172e-05 - mse: 2.8343e-05
Epoch 00028: val_loss improved from 0.00020 to 0.00017, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.5115e-05 - mse: 3.0230e-05 - val_loss: 1.6553e-04 - val_mse: 3.3106e-04
Epoch 29/50
145/150 [============================>.] - ETA: 0s - loss: 1.4089e-05 - mse: 2.8178e-05
Epoch 00029: val_loss improved from 0.00017 to 0.00016, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.4461e-05 - mse: 2.8921e-05 - val_loss: 1.5889e-04 - val_mse: 3.1778e-04
Epoch 30/50
144/150 [===========================>..] - ETA: 0s - loss: 1.5103e-05 - mse: 3.0207e-05
Epoch 00030: val_loss improved from 0.00016 to 0.00016, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.5549e-05 - mse: 3.1097e-05 - val_loss: 1.5603e-04 - val_mse: 3.1205e-04
Epoch 31/50
139/150 [==========================>...] - ETA: 0s - loss: 1.4528e-05 - mse: 2.9055e-05
Epoch 00031: val_loss did not improve from 0.00016
150/150 [==============================] - 1s 8ms/step - loss: 1.6304e-05 - mse: 3.2608e-05 - val_loss: 1.8720e-04 - val_mse: 3.7439e-04
Epoch 32/50
142/150 [===========================>..] - ETA: 0s - loss: 1.4269e-05 - mse: 2.8537e-05
Epoch 00032: val_loss improved from 0.00016 to 0.00015, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 9ms/step - loss: 1.4625e-05 - mse: 2.9250e-05 - val_loss: 1.4684e-04 - val_mse: 2.9369e-04
Epoch 33/50
142/150 [===========================>..] - ETA: 0s - loss: 1.2597e-05 - mse: 2.5194e-05
Epoch 00033: val_loss did not improve from 0.00015
150/150 [==============================] - 1s 8ms/step - loss: 1.3262e-05 - mse: 2.6524e-05 - val_loss: 1.8885e-04 - val_mse: 3.7770e-04
Epoch 34/50
146/150 [============================>.] - ETA: 0s - loss: 1.4380e-05 - mse: 2.8761e-05
Epoch 00034: val_loss improved from 0.00015 to 0.00014, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.4529e-05 - mse: 2.9058e-05 - val_loss: 1.4084e-04 - val_mse: 2.8168e-04
Epoch 35/50
144/150 [===========================>..] - ETA: 0s - loss: 1.2255e-05 - mse: 2.4510e-05
Epoch 00035: val_loss improved from 0.00014 to 0.00014, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.2630e-05 - mse: 2.5259e-05 - val_loss: 1.3570e-04 - val_mse: 2.7141e-04
Epoch 36/50
146/150 [============================>.] - ETA: 0s - loss: 1.2402e-05 - mse: 2.4805e-05
Epoch 00036: val_loss did not improve from 0.00014
150/150 [==============================] - 1s 8ms/step - loss: 1.2712e-05 - mse: 2.5423e-05 - val_loss: 1.3717e-04 - val_mse: 2.7434e-04
Epoch 37/50
148/150 [============================>.] - ETA: 0s - loss: 1.2014e-05 - mse: 2.4028e-05
Epoch 00037: val_loss improved from 0.00014 to 0.00013, saving model to tmp/ckeckpointer.ckpt
150/150 [==============================] - 1s 8ms/step - loss: 1.2155e-05 - mse: 2.4311e-05 - val_loss: 1.3075e-04 - val_mse: 2.6150e-04
Epoch 38/50
149/150 [============================>.] - ETA: 0s - loss: 1.2660e-05 - mse: 2.5319e-05
Epoch 00038: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.2697e-05 - mse: 2.5395e-05 - val_loss: 3.2183e-04 - val_mse: 6.4366e-04
Epoch 39/50
139/150 [==========================>...] - ETA: 0s - loss: 1.0822e-05 - mse: 2.1644e-05
Epoch 00039: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1924e-05 - mse: 2.3849e-05 - val_loss: 3.5369e-04 - val_mse: 7.0737e-04
Epoch 40/50
143/150 [===========================>..] - ETA: 0s - loss: 1.1230e-05 - mse: 2.2460e-05
Epoch 00040: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1633e-05 - mse: 2.3266e-05 - val_loss: 2.4469e-04 - val_mse: 4.8938e-04
Epoch 41/50
140/150 [===========================>..] - ETA: 0s - loss: 1.2174e-05 - mse: 2.4348e-05
Epoch 00041: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.2546e-05 - mse: 2.5092e-05 - val_loss: 2.2481e-04 - val_mse: 4.4961e-04
Epoch 42/50
141/150 [===========================>..] - ETA: 0s - loss: 1.0771e-05 - mse: 2.1541e-05
Epoch 00042: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1201e-05 - mse: 2.2402e-05 - val_loss: 1.9641e-04 - val_mse: 3.9283e-04
Epoch 43/50
150/150 [==============================] - ETA: 0s - loss: 1.1375e-05 - mse: 2.2750e-05
Epoch 00043: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1375e-05 - mse: 2.2750e-05 - val_loss: 5.3345e-04 - val_mse: 0.0011
Epoch 44/50
147/150 [============================>.] - ETA: 0s - loss: 1.0875e-05 - mse: 2.1749e-05
Epoch 00044: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.0975e-05 - mse: 2.1949e-05 - val_loss: 3.9328e-04 - val_mse: 7.8657e-04
Epoch 45/50
144/150 [===========================>..] - ETA: 0s - loss: 1.0671e-05 - mse: 2.1343e-05
Epoch 00045: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1074e-05 - mse: 2.2147e-05 - val_loss: 3.3973e-04 - val_mse: 6.7945e-04
Epoch 46/50
143/150 [===========================>..] - ETA: 0s - loss: 1.0369e-05 - mse: 2.0738e-05
Epoch 00046: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.0733e-05 - mse: 2.1465e-05 - val_loss: 3.1263e-04 - val_mse: 6.2526e-04
Epoch 47/50
149/150 [============================>.] - ETA: 0s - loss: 1.1758e-05 - mse: 2.3515e-05
Epoch 00047: val_loss did not improve from 0.00013
150/150 [==============================] - 1s 8ms/step - loss: 1.1775e-05 - mse: 2.3549e-05 - val_loss: 1.3305e-04 - val_mse: 2.6610e-04

model.load_weights(filename)
[ ]
model.load_weights(filename)
<tensorflow.python.training.tracking.util.CheckpointLoadStatus at 0x7f3cdd729518>
pred = model.predict(test_data)
pred.shape
(1180, 1)

plt.figure(figsize=(12, 9))
plt.plot(np.asarray(y_test)[20:], label='actual')
plt.plot(pred, label='prediction')
plt.legend()
plt.show()
