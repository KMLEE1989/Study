import numpy as np
import pandas as pd

path = "../_data/SamsungK/"

dataset1 =pd.read_csv(path + '삼성전자.csv', thousands=',')
dataset2 = pd.read_csv(path+ '키움증권.csv', thousnads=',')

print(dataset1)
#print(dataset2)


# (range(893, 1120), axis=0)
# x = dataset.drop(['일자','종가',"Unnamed: 6",'전일비','등락률','금액(백만)','신용비','개인','기관','외인(수량)','외국계','프로그램', '외인비'], axis=1) # axis=1 컬럼 삭제할 때 필요함

# y = dataset['종가']
# print(x,y)
# size = 20
# def split_x(data, size):
#     aaa = []
#     for i in range(len(data) - size +1): 
#         subset = data[i : (i+size)]       
#         aaa.append(subset)                  
#     return np.array(aaa)
# x = split_x(x,size)
# y = split_x(y,size)
# print(x.columns, x.shape)  # (1060, 13)
  # (1060, 4) (1060,)

# x = x.to_numpy()

# x = x.head(10)
# y = y.head(20)

# x_train, x_test, y_train, y_test = train_test_split(x,y,
#         train_size =0.7, shuffle=True, random_state = 42)

# print(x.shape,y.shape)  # (874, 20, 4) (874, 20)
# x_train = x_train.values.reshape(1120,4,1)
# x_test = x_test.values.reshape(1120,4,1)


# print(x_train.shape, x_test.shape)  # (954, 13) (106, 13)
# print(y_train.shape, y_test.shape)  # (954,) (106,)
# x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],1) 
# x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],1)

# print(x_train.shape, x_test.shape) # (954, 13, 1) (106, 13, 1)

#2. 모델구성
# model = Sequential()
# model.add(LSTM(32,activation='relu',input_shape = (20,4)))
# # model.add(Dense(130))
# model.add(Dense(100))
# model.add(Dense(130))
# model.add(Dense(80))
# model.add(Dense(5))
# model.add(Dense(1))

# model.summary()

#3. 컴파일, 훈련
# model.compile(loss='mae', optimizer = 'adam')
 
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# # es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
# #                    verbose=1, restore_best_weights=False)
# # mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
# #                        filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
# model.fit(x_train, y_train, epochs=200, batch_size=2,
#           validation_split=0.3)

# # model.save('../_test/_save/kovt.h5')

# #4. 평가, 예측
# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

# y_pred = model.predict(x_test)
# print(y_pred)


# dataset=pd.read_csv(path1+'삼성전자.csv',thousands=',') 
# dataset = dataset.drop(range(893, 1120), axis=0)

# x=dataset.drop(['일자','종가', '전일비','Unnamed: 6','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '등락률','외인비'],axis=1)
# y=dataset['종가']

# print(x,y)

# size=20
# def split_x(data,size):
#     aaa=[]
#     for i in range(len(data)-size+1):
#         subset=data[i: (i+size)]
#         aaa.append(subset)
#     return np.array(aaa)
# x=split_x(x,size)
# y=split_x(y,size)

# print(x.columns, x.shape)  #Index(['시가', '고가', '저가', '거래량'], dtype='object') #(893, 4)

# x_train, x_test, y_train, y_test=train_test_split(x,y,train_size=0.7, shuffle=True,random_state=66)

# print(x.shape, y.shape) # (874, 20, 4) (874, 20)

# model = Sequential()
# model.add(LSTM(32,activation='relu',input_shape = (20,4)))
# # model.add(Dense(130))
# model.add(Dense(100))
# model.add(Dense(130))
# model.add(Dense(80))
# model.add(Dense(5))
# model.add(Dense(1))

# model.compile(loss='mae', optimizer = 'adam')

# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# # es = EarlyStopping(monitor='val_loss', patience=50, mode='auto',
# #                    verbose=1, restore_best_weights=False)
# # mcp = ModelCheckpoint (monitor = 'val_loss', mode = 'min', verbose = 1, save_best_only=True,
# #                        filepath = '../Study/_ModelCheckPoint/kovt.hdf5')
# model.fit(x_train, y_train, epochs=200, batch_size=2, validation_split=0.3)

# loss = model.evaluate(x_test,y_test)
# print("loss : ",loss)

# y1_pred = model.predict(x_test)
# print('예측가격:',y1_pred)

# path2 = "../_data/SamsungK/Kium/"
# datasets2=pd.read_csv(path2+'키움증권.csv', thousands=',') 

# print(datasets1)  #[1120 rows x 17 columns]
# print(datasets1.shape)  #(1120, 17)

# print(datasets2)  #[1060 rows x 17 columns]
# print(datasets2.shape)  #(1060, 17)

# def split_xy5(dataset, time_steps, y_column):
#     x, y = list(), list()
#     for i in range(len(dataset)):
#         x_end_number = i + time_steps
#         y_end_number = x_end_number + y_column # 수정

#         if y_end_number > len(dataset):  # 수정
#             break
#         tmp_x = dataset[i:x_end_number, :]  # 수정
#         tmp_y = dataset[x_end_number:y_end_number, 3]    # 수정
#         x.append(tmp_x)
#         y.append(tmp_y)
#     return np.array(x), np.array(y)
# x1, y1 = split_xy5(datasets1, 5, 1) 
# x2, y2 = split_xy5(datasets2, 5, 1) 
# print(x2[0,:], "\n", y2[0])
# print(x2.shape)
# print(y2.shape)

# from sklearn.model_selection import train_test_split
# # from sklearn.model_selection import cross_val_score
# x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, random_state=1, test_size = 0.3)
# x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, random_state=2, test_size = 0.3)

# # x1=datasets1
# # # x2=datasets2

# x1=x1.drop(['일자','종가', '전일비','Unnamed: 6','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '등락률','외인비'], axis=1)
# x1=x1.drop(range(893,1060),axis=0)
# # x1_train=x1
# # y1=datasets1['종가']


# # # x2=x2.drop(['일자', '전일비','Unnamed: 6','금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '등락률','외인비'], axis=1)
# x2=x2.drop(range(893,1060),axis=0)
# # # x2_train=x2
# # # y2=datasets2['종가']


# # print(x1.columns, x1.shape) #Index(['시가', '고가', '저가', '종가', '거래량'], dtype='object') (1060, 5)
# # # print(x2.columns, x2.shape) #Index(['시가', '고가', '저가', '종가', '거래량'], dtype='object') (1160, 5)
# # x1.head(5)
# # y1.head(5)


# # x1=x1.to_numpy()
# # y1=y1.to_numpy()


# # x1_train, x1_test, y1_train, y1_test= train_test_split(x1,y1,
# #                 train_size=0.7, shuffle=True, random_state=66)


# # model = Sequential() 
# # model.add(Dense(20, input_dim=5))
# # model.add(Dense(17))
# # model.add(Dense(14))
# # model.add(Dense(3))
# # model.add(Dense(2))
# # model.add(Dense(1))


# # model.compile(loss='mse', optimizer='adam')
# # model.fit(x1_train, y1_train, epochs=800, batch_size=13)


# # loss=model.evaluate(x1_test, y1_test) 
# # print('loss :', loss)

# # y1_predict= model.predict(x1_test)
# # print('예측가격:', y1_predict)


# # # x1_train, x1_test,x2_train,x2_test,y1_train,y1_test, y2_train, y2_test, = train_test_split(x1,x2,y1,y2, train_size=0.7, shuffle=True, random_state=66)

# # # print(x1_train.shape, x1_test.shape)  
# # # print(x2_test.shape, x2_test.shape) 
# # # print(y1_train.shape, y1_test.shape) 
# # # print(y2_train.shape, y2_test.shape)


# # """
# # x1_train, x1_test,y1_train,y1_test, y2_train, y2_test, = train_test_split(x1,y1,y2, train_size=0.7, shuffle=True, random_state=66)
# # # x1_test, x1_val, y_test, y_val=train_test_split(x1_test,y_test,train_size=0.7, shuffle=True, random_state=66)
# # # x2_test, x2_val, y_test, y_val=train_test_split(x2_test, y_test,train_size=0.7, shuffle=True, random_state=66)

# # print(x1_train.shape, x1_test.shape)  #(70, 2) (30, 2)
# # print(y1_train.shape, y1_test.shape)  #(70,) (30,)
# # print(y2_train.shape, y2_test.shape)  #(70,) (30,)
# # print(y3_train.shape, y3_test.shape)  #(70,) (30,)

# # #2. 모델구성
# # from tensorflow.keras.models import Model
# # from tensorflow.keras.layers import Dense, Input

# # #2-1 모델1
# # input1=Input(shape=(2,))
# # dense1=Dense(5, activation='relu', name='dense1')(input1)
# # dense2=Dense(7, activation='relu', name='dense2')(dense1)
# # dense3=Dense(7, activation='relu', name='dense3')(dense2)
# # output1=Dense(7, activation='relu', name='dense4')(dense3)
# # """
# # '''
# # #2-1 모델1
# # input2=Input(shape=(3,))
# # dense11=Dense(5, activation='relu', name='dense11')(input2)
# # dense12=Dense(10, activation='relu', name='dense12')(dense11)
# # dense13=Dense(7, activation='relu', name='dense13')(dense12)
# # dense14=Dense(7, activation='relu', name='dense14')(dense13)
# # output2=Dense(5, activation='relu', name='output2')(dense14)
# # '''
# # # from tensorflow.keras.layers import concatenate, Concatenate

# # # merge1 = concatenate([output1])

# # # #2-3 output 모델1
# # # output21 = Dense(7)(merge1)
# # # output22 = Dense(11)(output21)
# # # output23 = Dense(11, activation='relu')(output22)
# # # last_output1 = Dense(1)(output23)


# # # #2-3 output 모델1
# # # output31 = Dense(7)(merge1)
# # # output32 = Dense(21)(output31)
# # # output33 = Dense(21, activation='relu')(output32)
# # # output34 = Dense(11, activation='relu')(output33)
# # # last_output2 = Dense(1)(output34)

# # # output41=Dense(7)(merge1)
# # # output42=Dense(21)(output41)
# # # output43=Dense(21,activation='relu')(output42)
# # # output44=Dense(11,activation='relu')(output43)
# # # last_output3=Dense(1)(output44)

# # # # merge2 = Dense(10, activation='relu')(merge1)
# # # # merge3 = Dense(7)(merge2)
# # # # last_output=Dense(1)(merge3)

# # # model = Model(inputs=[input1], outputs=[last_output1, last_output2, last_output3])

# # # # 3. 훈련
# # # model.compile(loss='mse', optimizer='adam', metrics=['mse'])
# # # model.fit([x1_train],[y1_train, y2_train, y3_train], epochs=100, batch_size=8, validation_data=([x1],[y1, y2,y3]), verbose=1) 

# # # # model.compile(loss='mse', optimizer='adam', matrics=['mse'])
# # # # model.fit([x1_train, x2_train], [y1_train, y2_train], epochs=100, batch_size=8, verbose=1)

# # # results=model.evaluate([x1_test], [y1_test, y2_test,y3_test])


# # #print(x1.columns)
# # # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
# # #        '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
# # #       dtype='object')

# # # print(x2.columns)
# # # Index(['일자', '시가', '고가', '저가', '종가', '전일비', 'Unnamed: 6', '등락률', '거래량',
# # #        '금액(백만)', '신용비', '개인', '기관', '외인(수량)', '외국계', '프로그램', '외인비'],
# # #       dtype='object')




# # # path = "../_data/kaggle/bike/"

# # # train = pd.read_csv(path+'train.csv')
# # # #print(train)  #(10886, 12)
# # # print(train.shape)
# # #pd.to_datetime(samsung['일자'], format='%Y%m%d')


# # '''
# # # path = "../_data/dacon/heart_disease/"
# # df_price.describe()

# # pd.to_datetime(df_price['일자'], format='%Y%m%d')
# # # 0      2020-01-07
# # # 1      2020-01-06
# # # 2      2020-01-03
# # # 3      2020-01-02
# # # 4      2019-12-30

# # df_price['일자'] = pd.to_datetime(df_price['일자'], format='%Y%m%d')
# # df_price['연도'] =df_price['일자'].dt.year
# # df_price['월'] =df_price['일자'].dt.month
# # df_price['일'] =df_price['일자'].dt.day

# # [stock-data-01](../../../Downloads/07-CHROME_DOWNLOAD_200213/0215-blog/stock-data-01.pngdf = df_price.loc[df_price['연도']>=1990]

# # plt.figure(figsize=(16, 9))
# # sns.lineplot(y=df['종가'], x=df['일자'])
# # plt.xlabel('time')
# # plt.ylabel('price')

# # from sklearn.preprocessing import MinMaxScaler

# # scaler = MinMaxScaler()
# # scale_cols = ['시가', '고가', '저가', '종가', '거래량']
# # df_scaled = scaler.fit_transform(df[scale_cols])

# # df_scaled = pd.DataFrame(df_scaled)
# # df_scaled.columns = scale_cols

# # print(df_scaled)

# # train = df_scaled[:-TEST_SIZE]
# # test = df_scaled[-TEST_SIZE:]

# # def make_dataset(data, label, window_size=20):
# #     feature_list = []
# #     label_list = []
# #     for i in range(len(data) - window_size):
# #         feature_list.append(np.array(data.iloc[i:i+window_size]))
# #         label_list.append(np.array(label.iloc[i+window_size]))
# #     return np.array(feature_list), np.array(label_list)
    
# #     feature_cols = ['시가', '고가', '저가', '거래량']
# # label_cols = ['종가']

# # train_feature = train[feature_cols]
# # train_label = train[label_cols]

# # # train dataset
# # train_feature, train_label = make_dataset(train_feature, train_label, 20)

# # # train, validation set 생성
# # from sklearn.model_selection import train_test_split
# # x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2)

# # x_train.shape, x_valid.shape
# # # ((6086, 20, 4), (1522, 20, 4))

# # # test dataset (실제 예측 해볼 데이터)
# # test_feature, test_label = make_dataset(test_feature, test_label, 20)
# # test_feature.shape, test_label.shape
# # # ((180, 20, 4), (180, 1))

# # from keras.models import Sequential
# # from keras.layers import Dense
# # from keras.callbacks import EarlyStopping, ModelCheckpoint
# # from keras.layers import LSTM

# # model = Sequential()
# # model.add(LSTM(16, 
# #                input_shape=(train_feature.shape[1], train_feature.shape[2]), 
# #                activation='relu', 
# #                return_sequences=False)
# #           )
# # model.add(Dense(1))

# # model.compile(loss='mean_squared_error', optimizer='adam')
# # early_stop = EarlyStopping(monitor='val_loss', patience=5)
# # filename = os.path.join(model_path, 'tmp_checkpoint.h5')
# # checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

# # history = model.fit(x_train, y_train, 
# #                     epochs=200, 
# #                     batch_size=16,
# #                     validation_data=(x_valid, y_valid), 
# #                     callbacks=[early_stop, checkpoint])

# # # ...
# # # ...

# # # Epoch 00015: val_loss did not improve from 0.00002
# # # Epoch 16/200
# # # 6086/6086 [==============================] - 12s 2ms/step - loss: 3.1661e-05 - val_loss: 4.1063e-05

# # # Epoch 00016: val_loss did not improve from 0.00002
# # # Epoch 17/200
# # # 6086/6086 [==============================] - 13s 2ms/step - loss: 2.4644e-05 - val_loss: 4.0085e-05

# # # Epoch 00017: val_loss did not improve from 0.00002
# # # Epoch 18/200
# # # 6086/6086 [==============================] - 13s 2ms/step - loss: 2.2936e-05 - val_loss: 2.4692e-05

# # # Epoch 00018: val_loss did not improve from 0.00002

# # # weight 로딩
# # model.load_weights(filename)

# # # 예측
# # pred = model.predict(test_feature)

# # plt.figure(figsize=(12, 9))
# # plt.plot(test_label, label='actual')
# # plt.plot(pred, label='prediction')
# # plt.legend()
# # plt.show()

# # '''