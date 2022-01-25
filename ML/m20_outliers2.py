#실습 다차원의 아웃라이어가 출력 되도록
from matplotlib.pyplot import boxplot
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import numpy as np

'''
aaa= np.array([[1,2,-20,4,5,6,7,8,30,100,500,12,13],
              [100,200,3,400,500,600,7,800,900,190,1001,1002,99]])

#(2,13) -> (13,2)

aaa=np.transpose(aaa) #(13,2)

def outliers(data_out):
    quartile_1, q2, quartile_3 = np.percentile(data_out,[25,50,75])
    
    print("1사분위 : ", quartile_1)
    print("q2: ", q2)
    print("3사분위: ", quartile_3)
    iqr = quartile_3-quartile_1
    print("iqr : ", iqr)
    lower_bound = quartile_1 - (iqr*1.5)
    upper_bound = quartile_3 +(iqr*1.5)
    return np.where((data_out>upper_bound)|  #또는
                    (data_out<lower_bound))
    
outliers_loc = outliers(aaa)
print("이상치의 위치 : ", outliers_loc)
    
# 시각화
# 실습
# boxplot 

plt.boxplot(aaa)
plt.show()
'''

import numpy as np

aaa = np.array( [
                [1,2, -20, 4, 5, 6, 7, 8, 30, 100, 500, 12, 13],
                [100, 200, 3, 400, 500, 600, 7, 800, 900, 190, 1001, 1002, 99]
                ] ) # 2행 13열

# (2, 13) => (13, 2)
aaa = np.transpose(aaa)

print(aaa)
def outliers(data_out):
    try:
        i=0
        while True:
            # 반복설정
            i=i+1

            if data_out[:,i-1:i] is not None :
               quantile_1, q2, quantile_3 = np.percentile(data_out[:,i-1:i], [25,50,75])
               print(i,"행")
               print("1사분위 : ", quantile_1)
               print("q2 : ", q2)
               print("3사분위 : ", quantile_3)
               
               iqr = quantile_3 - quantile_1
               print("iqr : ", iqr)
               print("\n")
               lower_bound = quantile_1 - (iqr * 1.5)
               upper_bound = quantile_3 + (iqr * 1.5)

            else:
                return np.where((data_out[:,i-1:i] > upper_bound) |        #  이 줄 또는( | )
                            (data_out[:,i-1:i] < lower_bound))         #  아랫줄일 경우 반환
    except Exception:
        pass

print(outliers(aaa))