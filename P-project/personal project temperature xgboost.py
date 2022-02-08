import xgboost
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score
import pandas as pd
import numpy as np

path = "../_data/개인프로젝트/CSV/"

df1=pd.read_csv(path+'기온 데이터.csv',thousands=',')

