import yfinance as yf
import matplotlib.pyplot as plt

#삼성전자 주가 다운로드(2020-01-01 ~ 2021-12-01)
samsung = yf.download('005930.KS', #삼성전자주 코드
start='2020-01-01', end='2021-12-01')

# # [close == 종가]
samsung = samsung[['Close']]
samsung.reset_index(inplace=True)
samsung = samsung.rename(columns = {'Close' : 'Price'})
print(samsung.head(3))

samsung.plot(x='Date', y='Price', kind='line')
plt.show()

from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# ARIMA model 생성
model = ARIMA(samsung.Price.values, order=(2,1,2)) # order 안의 파라미터는 (AR, Difference, MA)
fit = model.fit()
fit.summary()