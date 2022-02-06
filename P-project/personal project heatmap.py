from seaborn.matrix import heatmap
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
import seaborn as sns
import matplotlib.pyplot as plt
 

path = "../_data/개인프로젝트/CSV/"

dft=pd.read_csv(path+'통합 XGBOOST.csv',thousands=',')

#print(dft.info())

dft=dft.drop(['지점','DATE','MAX TEMP(℃)','MIN TEMP(℃)','INSPECTION'], axis=1)

print(dft.info())

colormap=plt.cm.PuBu
plt.figure(figsize=(20,20))
plt.title("TEMP Correlation of Fteatures", y=1.00, size=15)
sns.heatmap(dft.astype(float).corr(), linewidths=0.08, vmax=1.0, square=True, cmap=colormap, linecolor="white", annot=True, annot_kws={"size":6})

plt.show()
