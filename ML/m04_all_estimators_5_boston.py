from sklearn import datasets
from sklearn.utils import all_estimators  #회기모델regressor은 r2스코어 accuracy= classifier
from sklearn.metrics import accuracy_score, r2_score
from sklearn.datasets import load_boston
import warnings 
warnings.filterwarnings('ignore')

datasets=load_boston()
x=datasets.data
y=datasets.target

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)

#allAlgorithm = all_estimators(type_filter='classifier')
allAlgorithm = all_estimators(type_filter='regressor')


print("allAlgorithms : ", allAlgorithm)
print("모델의 갯수: ", len(allAlgorithm)) #3대장이 없어

for (name, algorithm) in allAlgorithm:
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률 : ', r2)

    except:
        continue
    

# ARDRegression 의 정답률 :  0.8119016106669674
# AdaBoostRegressor 의 정답률 :  0.9010953675400355
# BaggingRegressor 의 정답률 :  0.9082051600737113
# BayesianRidge 의 정답률 :  0.8119880571377844
# CCA 의 정답률 :  0.7913477184424631
# DecisionTreeRegressor 의 정답률 :  0.8165588634586637
# DummyRegressor 의 정답률 :  -0.0005370164400797517
# ElasticNet 의 정답률 :  0.16201563080833714
# ElasticNetCV 의 정답률 :  0.8113737663385279
# ExtraTreeRegressor 의 정답률 :  0.837015238601132
# ExtraTreesRegressor 의 정답률 :  0.9340561065490484
# GammaRegressor 의 정답률 :  0.19647920570298638
# GaussianProcessRegressor 의 정답률 :  -1.5789586748045834
# GradientBoostingRegressor 의 정답률 :  0.9457468055664106
# HistGradientBoostingRegressor 의 정답률 :  0.9323326124661162
# HuberRegressor 의 정답률 :  0.7958372970870966
# KNeighborsRegressor 의 정답률 :  0.8265307833211177
# KernelRidge 의 정답률 :  0.8032549585020756
# Lars 의 정답률 :  0.7746736096721606
# LarsCV 의 정답률 :  0.7981576314184021
# Lasso 의 정답률 :  0.242592140544296
# LassoCV 의 정답률 :  0.8125908596954046
# LassoLars 의 정답률 :  -0.0005370164400797517
# LassoLarsCV 의 정답률 :  0.8127604328474284
# LassoLarsIC 의 정답률 :  0.8131423868817644
# LinearRegression 의 정답률 :  0.8111288663608667
# LinearSVR 의 정답률 :  0.708268196924303
# MLPRegressor 의 정답률 :  0.45222824526088634
# NuSVR 의 정답률 :  0.6254681434531
# OrthogonalMatchingPursuit 의 정답률 :  0.582761757138145
# OrthogonalMatchingPursuitCV 의 정답률 :  0.78617447738729
# PLSCanonical 의 정답률 :  -2.231707974142574
# PLSRegression 의 정답률 :  0.8027313142007888
# PassiveAggressiveRegressor 의 정답률 :  0.7086317725350348
# PoissonRegressor 의 정답률 :  0.6749600710148136
# QuantileRegressor 의 정답률 :  -0.020280478327197038
# RANSACRegressor 의 정답률 :  0.6132339925116463
# RadiusNeighborsRegressor 의 정답률 :  0.41191760158788593
# RandomForestRegressor 의 정답률 :  0.9264225060168108
# Ridge 의 정답률 :  0.8087497007195746
# RidgeCV 의 정답률 :  0.8116598578372443
# SGDRegressor 의 정답률 :  0.8231765438845414
# SVR 의 정답률 :  0.6597910766772523
# TheilSenRegressor 의 정답률 :  0.781655938324942
# TransformedTargetRegressor 의 정답률 :  0.8111288663608667
# TweedieRegressor 의 정답률 :  0.19473445117356525
