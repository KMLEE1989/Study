from xgboost import XGBClassifier, XGBRegressor
from sklearn.datasets import fetch_california_housing, load_boston
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import RobustScaler, MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import PowerTransformer

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
import time
# import warnings
# warnings.filterwarnings('ignore')

#datasets = fetch_california_housing()
datasets = load_boston()
x=datasets.data
y=datasets['target']

# print(x.shape, y.shape) (20640, 8) (20640,)

x_train, x_test,y_train,y_test= train_test_split(x,y,shuffle=True, random_state=66, train_size=0.8)

scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test=scaler.transform(x_test)

#model=XGBRegressor()
model=XGBRegressor(n_jobs=-1, verbose=1, n_estimators=1000,
    learning_rate=0.025,               
    max_depth=4,
    min_child_weight = 10,
    subsample=0.5,
    colsample_bytree = 1, 
    reg_alpha = 1,              #규제  L1
    reg_lambda=0,               # 규제 L2
    )

start = time.time()
model.fit(x_train, y_train, verbose=1,
          eval_set=[(x_train,y_train),(x_test,y_test)],
          eval_metric='rmse',
          early_stopping_rounds=10) #rmse, mae, logloss, error
end = time.time()

print("걸린시간 : ", end - start)

results=model.score(x_test, y_test)

print("results : ", round(results,5))

y_predict = model.predict(x_test)

r2= r2_score(y_test, y_predict)

print("r2 : ", round(r2,4))

# results :  0.843390038548427
# r2 :  0.843390038548427

print("===================================")
hist = model.evals_result()

loss1 = hist.get('validation_0').get('rmse')
loss2 = hist.get('validation_1').get('rmse')
# plt.plot(loss1, 'r--', label="training loss")
# plt.plot(loss2, 'b--', label="test loss")
# plt.grid()
# plt.legend()
# plt.show()

# loss1 = hist.get('validation_0').get('mae')
# loss2 = hist.get('validation_1').get('mae')
plt.figure(figsize=(9,5)) # 판을 깔다.
plt.plot(loss1, marker= '.', c='red', label='loss') # 선, 점을 그리다.
plt.plot(loss2, marker= '.', c='blue', label='val_loss') 
plt.grid() # 격자를 보이게
plt.title('loss') # 제목
plt.ylabel('loss') 
plt.xlabel('epoch')
plt.legend(loc='upper right')
plt.show()


# 걸린시간 :  9.32486605644226
# results :  0.8566291699938181
# r2 :  0.8566291699938181

# BOSTON
# 걸린시간 :  0.5529332160949707
# results :  0.9313449710981906
# r2 :  0.9313449710981906