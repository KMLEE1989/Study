import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer, load_wine, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import warnings
warnings.filterwarnings(action='ignore')

# LDA) y의 target값을 넣어서 전처리 해준다! lable값에 맞추어 차원 축소

#1. 데이터
#datasets = load_iris()
#datasets = load_breast_cancer()
#datasets = load_wine()
datasets = fetch_covtype()

x = datasets.data
y = datasets.target
print("LDA 전 : ", x.shape)  

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=66, shuffle=True, stratify=y)

# stratify: 균등하게 빼주는 것 / 분류모델에 주로 사용(고르게)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


# pca = PCA(n_components=8)
lda = LinearDiscriminantAnalysis() 
# x = pca.fit_transform(x)         

lda.fit(x_train, y_train) ### y_train이라는 답지를 보고 x_train의 컬럼을 줄이겠다(=지도학습과 비슷) ### PCA는 x데이터로 x컬럼을 줄이는것(참고자료 없이) 그래서 비지도학습과 비슷
x_train = lda.transform(x_train)   ## 그래서 y_train은 transform해줄 필요x
x_test = lda.transform(x_test)

print("LDA 후: ", x_train.shape)

#2. 모델
from xgboost import XGBClassifier
model = XGBClassifier()

#3. 훈련
import time
start = time.time()
#model.fit(x_train, y_train, eval_metric='error') # 이진분류
model.fit(x_train, y_train, eval_metric='merror')  # 다중분류
# model.fit(x_train, y_train)

end = time.time()

#4. 평가, 예측
results = model.score(x_test, y_test)
print('results:', results)
print('걸린 시간: ', end-start)

#IRIS
# LDA 전 :  (150, 4)
# LDA 후:  (120, 2)
# results: 1.0
# 걸린 시간:  0.06726789474487305

#Cancer
# LDA 전 :  (569, 30)
# LDA 후:  (455, 1)
# results: 0.9473684210526315
# 걸린 시간:  0.05139923095703125

#Wine
# LDA 전 :  (178, 13)
# LDA 후:  (142, 2)
# results: 1.0
# 걸린 시간:  0.06034731864929199

# Fetch_cov()
# LDA 전 :  (581012, 54)
# LDA 후:  (464809, 6)
# results: 0.7878109859470065
# 걸린 시간:  82.02236199378967
