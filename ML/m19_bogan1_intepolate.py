# 결측치 처리
#1. 행또는 열 삭제
#2. 임의의값 
# fillna -  0,ffill, bfill, 중위값, 평균값.... 76767
#3. 보간 - interpolate  
#4. 모델링 = predict
#5. 부스팅계열 = 통상 결측치, 이상치에 대해 자유롭다. 믿거나말거나 ^^


from turtle import pd
import pandas as pd
import numpy as np
from datetime import datetime

dates = ['1/24/2022', ' 1/25/2022', '1/26/2022', '1/27/2022','1/28/2022',]
dates=pd.to_datetime(dates)
print(dates)

ts=pd.Series([2, np.nan, np.nan, 8, 10], index=dates)
print(ts)

ts=ts.interpolate()
print(ts)

