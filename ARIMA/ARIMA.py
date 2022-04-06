from ntpath import join
import FinanceDataReader as fdr
import pandas as pd
import matplotlib.pyplot as plt
import plot
import numpy as np
from pmdarima.arima import ndiffs
import pmdarima as pm
import os
from matplotlib import font_manager, rc
import matplotlib as mpl
import csv
import matplotlib.font_manager as fm
import matplotlib_inline

font_path = "C:/Windows/Fonts/HMFMMUEX.TTC"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)
fontprop=fm.FontProperties(fname=font_path, size=18)

print(fdr.__version__)


# df_krx = fdr.StockListing('KRX')
# print(df_krx.head())
# print(len(df_krx))

woori = fdr.DataReader('041190', '2022')
print(woori.head(100))

woori['Close'].plot()
plt.title('<우리기술투자 주식 가격>', font=fontprop)
plt.ylabel('가격', font=fontprop)
plt.xlabel('날짜 2주 간격', font=fontprop)
plt.show()

# from statsmodels.tsa.arima.model import ARIMA
# import statsmodels.api as sm

# woori = woori[['Close']]
# woori.reset_index(inplace=True)
# woori = woori.rename(columns = {'Close': 'Price'})
# print(woori.head(100))

# model = ARIMA(woori.Price.values, order=(0,1,1))
# fit = model.fit()
# print(fit.summary())

