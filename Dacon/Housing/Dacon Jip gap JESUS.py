
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# 데이터 전처리
datasets = pd.read_csv('D:/_data/dacon/housing/train.csv')
test = pd.read_csv('D:/_data/dacon/housing/test.csv')
submit = pd.read_csv('D:/_data/dacon/housing/sample_submission.csv')

x = datasets.drop(columns=['id', 'target'])
y = datasets['target']
test = test.drop(columns=['id'])


# 라벨인코딩
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x['Exter Qual'] = le.fit_transform(x['Exter Qual'])
x['Kitchen Qual'] = le.fit_transform(x['Kitchen Qual'])
x['Bsmt Qual'] = le.fit_transform(x['Bsmt Qual'])

test['Exter Qual'] = le.fit_transform(test['Exter Qual'])
test['Kitchen Qual'] = le.fit_transform(test['Kitchen Qual'])
test['Bsmt Qual'] = le.fit_transform(test['Bsmt Qual'])

print(x.shape, y.shape)  # (1350, 13) (1350,)
# print(datasets.corr())

# train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66)

# parameter
#1. for XGBooster
parameter = [
    {'xg__n_estimators':[100, 200, 300], 'xg__learning_rate':[0.1, 0.3, 0.001, 0.01, 0.05],
     'xg__max_depth':[4, 5, 6], 'xg__colsample_bytree':[0.6, 0.9, 1], 'xg__eval_metric':['rmse']}]

#2. for CatBooster 
# parameter = [
#     {'cb__n_estimators':(100, 200, 300),
#      'cb__bagging_temperature':(0, 1000),
#      'cb__depth':(5, 6, 7, 8),
#      'cb__learning_rate':(0.1, 0.5, 0.01, 0.025, 0.001),
#      'cb__min_data_in_leaf':(1, 6),
#      'cb__border_count':(5, 255),
#      'cb__verbose':(0),
#      }
# ]

#2. 모델 구성
from sklearn.pipeline import Pipeline
pipe = Pipeline([('st', StandardScaler()), ('xg', XGBRegressor())])
model = GridSearchCV(pipe, parameter, cv=5, n_jobs=2)

#3. 훈련
model.fit(x_train, y_train)


#4. 예측, 평가
# y_pred = np.round(model.predict(x_test), 0).astype(int)
y_pred = model.predict(x_test)

def NMAE(true, pred):
    mae = np.mean(np.abs(true-pred))
    score = mae / np.mean(np.abs(true))
    return score
print(NMAE(y_test, y_pred))


# 제출
test_pred = model.predict(test)
submit['target'] = test_pred
submit.to_csv("D:/_data/dacon/housing/submit.csv", index=False)

