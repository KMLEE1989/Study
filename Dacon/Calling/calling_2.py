import pandas as pd
import random
import os
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
from sklearn.pipeline import make_pipeline, Pipeline
from catboost import CatBoostClassifier

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

seed_everything(42) # Seed 고정

train = pd.read_csv('C:/KM LEE/calling/train.csv')
test = pd.read_csv('C:/KM LEE/calling/test.csv')

