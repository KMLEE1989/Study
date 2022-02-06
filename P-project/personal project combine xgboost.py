import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import pandas as np
from xgboost.sklearn import XGBRegressor
import warnings
warnings.filterwarnings('ignore')
from matplotlib import font_manager, rc 
from xgboost import plot_importance
 

path = "../_data/개인프로젝트/CSV/"

dft=pd.read_csv(path+'통합 XGBOOST.csv',thousands=',')

print(dft.info())

dft_X=dft.drop(['DATE','지점','AVERAGE TEMP(℃)'],axis=1)
dft_Y=dft['AVERAGE TEMP(℃)']

dft_x_train, dft_x_test, dft_y_train, dft_y_test=train_test_split(dft_X, dft_Y, test_size=0.2)
xgb_model=xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsampel=0.75, colsample_bytree=1, max_depth=7)

print(len(dft_x_train), len(dft_x_test))
xgb_model.fit(dft_x_train, dft_y_train)

XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.08, max_delta_step=0, max_depth=7, min_child_weight=1,
             missing=None, n_estimators=100, n_jobs=1, nthread=None, objective='reg:linear', random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=True, subsample=0.75)

xgboost.plot_importance(xgb_model)
fig, ax = plt.subplots(figsize=(10,12))
plot_importance(xgb_model, ax=ax)

predictions=xgb_model.predict(dft_x_test)
predictions

r_sq=xgb_model.score(dft_x_train, dft_y_train)
print(r_sq)
print(explained_variance_score(predictions, dft_y_test))

plt.show()

