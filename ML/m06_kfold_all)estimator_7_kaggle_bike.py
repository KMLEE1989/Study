from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score,  r2_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, KFold, cross_val_score  
import pandas as pd, numpy as np

#1. 데이터
path = "../_data/kaggle/bike/"    

train = pd.read_csv(path + 'train.csv')
test_file = pd.read_csv(path + 'test.csv')  
submit_file = pd.read_csv(path + 'sampleSubmission.csv')

x = train.drop(['datetime', 'casual','registered', 'count'], axis=1) 
test_file = test_file.drop(['datetime'], axis=1)
y = train['count']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size= 0.8, shuffle = True, random_state = 66)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#2. 모델 구성
#allAlgorithms = all_estimators(type_filter = 'classifier')  # classifier에 대한 모든 측정기
#print("allAlgorithms: ", allAlgorithms)  # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수:  41


allAlgorithms = all_estimators(type_filter = 'regressor')  # regressor에 대한 모든 측정기

#print("allAlgorithms: ", allAlgorithms)  # [('ARDRegression', <class 'sklearn.linear_model._bayes.ARDRegression'>), ..]
#print("모델의 갯수: ", len(allAlgorithms))  # 모델의 갯수: 55

for (name, algorithm) in allAlgorithms:   # [('AdaBoostClassifier', <class 'sklearn.ensemble._weight_boosting.AdaBoostClassifier'>),...
    try:
        model = algorithm()
        model.fit(x_train, y_train)
        n_splits = 5
        kfold = KFold(n_splits = n_splits, shuffle = True, random_state = 66)
        scores = cross_val_score(model, x_train, y_train, cv = kfold)   # cv = kfold 이만큼 교차검증을 시키겠다. 
        
        y_predict = model.predict(x_test)
        r2 = r2_score(y_test, y_predict)
        print(name, '의 정답률: ', r2, round(np.mean(scores),4))
    except:                     
        #continue   
        print(name, '은 에러') 
        
# AdaBoostClassifier 의 정답률:  0.8888888888888888 0.8099
# BaggingClassifier 의 정답률:  1.0 0.9155
# BernoulliNB 의 정답률:  0.4166666666666667 0.3175
# CalibratedClassifierCV 의 정답률:  0.9722222222222222 0.9786
# CategoricalNB 의 정답률:  0.5 nan
# ClassifierChain 은 에러
# ComplementNB 의 정답률:  0.8611111111111112 0.8648
# DecisionTreeClassifier 의 정답률:  0.9166666666666666 0.8872
# DummyClassifier 의 정답률:  0.4166666666666667 0.3958
# ExtraTreeClassifier 의 정답률:  0.9722222222222222 0.8874
# ExtraTreesClassifier 의 정답률:  1.0 0.9714
# GaussianNB 의 정답률:  1.0 0.9786
# GaussianProcessClassifier 의 정답률:  1.0 0.9643
# GradientBoostingClassifier 의 정답률:  0.9722222222222222 0.8941
# HistGradientBoostingClassifier 의 정답률:  0.9722222222222222 0.9507
# KNeighborsClassifier 의 정답률:  1.0 0.9571
# LabelPropagation 의 정답률:  1.0 0.9645
# LabelSpreading 의 정답률:  1.0 0.9645
# LinearDiscriminantAnalysis 의 정답률:  1.0 0.9857
# LinearSVC 의 정답률:  0.9722222222222222 0.9786
# LogisticRegression 의 정답률:  1.0 0.9643
# LogisticRegressionCV 의 정답률:  0.9722222222222222 0.9643
# MLPClassifier 의 정답률:  0.9722222222222222 0.9645
# MultiOutputClassifier 은 에러
# MultinomialNB 의 정답률:  0.9444444444444444 0.8805
# NearestCentroid 의 정답률:  1.0 0.9574
# NuSVC 의 정답률:  1.0 0.9714
# OneVsOneClassifier 은 에러
# OneVsRestClassifier 은 에러
# OutputCodeClassifier 은 에러
# PassiveAggressiveClassifier 의 정답률:  0.9722222222222222 1.0
# QuadraticDiscriminantAnalysis 의 정답률:  0.9722222222222222 0.9788
# RadiusNeighborsClassifier 의 정답률:  0.9722222222222222 0.9222
# RandomForestClassifier 의 정답률:  1.0 0.9717
# RidgeClassifier 의 정답률:  1.0 0.9786
# RidgeClassifierCV 의 정답률:  0.9722222222222222 0.9786
# SGDClassifier 의 정답률:  0.9722222222222222 0.9714
# SVC 의 정답률:  1.0 0.9786
# StackingClassifier 은 에러
# VotingClassifier 은 에러
# PS D:\Study>  d:; cd 'd:\Study'; & 'C:\ProgramData\Anaconda3\envs\tf270gpu\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '51225' '--' 'd:\Study\ml\m06_kfold_all)estimator_5_boston.py' 
# ARDRegression 의 정답률:  0.8119016106669674 0.6843
# AdaBoostRegressor 의 정답률:  0.8932664957929117 0.7776
# BaggingRegressor 의 정답률:  0.9240002733220478 0.8063
# BayesianRidge 의 정답률:  0.8119880571377844 0.6841
# CCA 의 정답률:  0.7913477184424631 0.6351
# DecisionTreeRegressor 의 정답률:  0.8161248692612604 0.6964
# DummyRegressor 의 정답률:  -0.0005370164400797517 -0.0223
# ElasticNet 의 정답률:  0.16201563080833714 0.1313
# ElasticNetCV 의 정답률:  0.8113737663385279 0.6852
# ExtraTreeRegressor 의 정답률:  0.8078414448773091 0.631
# ExtraTreesRegressor 의 정답률:  0.936961005655862 0.8441
# GammaRegressor 의 정답률:  0.19647920570298638 0.1684
# GaussianProcessRegressor 의 정답률:  -1.5789586748045834 -1.8099
# GradientBoostingRegressor 의 정답률:  0.945936305536592 0.8546
# HistGradientBoostingRegressor 의 정답률:  0.9323326124661162 0.8317
# HuberRegressor 의 정답률:  0.7958372970870966 0.6666
# IsotonicRegression 은 에러
# KNeighborsRegressor 의 정답률:  0.8265307833211177 0.6519
# KernelRidge 의 정답률:  0.8032549585020756 0.5899
# Lars 의 정답률:  0.7746736096721606 0.6651
# LarsCV 의 정답률:  0.7981576314184021 0.6678
# Lasso 의 정답률:  0.242592140544296 0.2115
# LassoCV 의 정답률:  0.8125908596954046 0.6839
# LassoLars 의 정답률:  -0.0005370164400797517 -0.0223
# LassoLarsCV 의 정답률:  0.8127604328474284 0.684
# LassoLarsIC 의 정답률:  0.8131423868817644 0.6756
# LinearRegression 의 정답률:  0.8111288663608667 0.6822
# LinearSVR 의 정답률:  0.7088039639910066 0.5688
# MLPRegressor 의 정답률:  0.41688634689490245 0.1128
# MultiOutputRegressor 은 에러
# MultiTaskElasticNet 은 에러
# MultiTaskElasticNetCV 은 에러
# MultiTaskLasso 은 에러
# MultiTaskLassoCV 은 에러
# NuSVR 의 정답률:  0.6254681434531 0.515
# OrthogonalMatchingPursuit 의 정답률:  0.582761757138145 0.5181
# OrthogonalMatchingPursuitCV 의 정답률:  0.78617447738729 0.6354
# PLSCanonical 의 정답률:  -2.231707974142574 -2.2912
# PLSRegression 의 정답률:  0.8027313142007888 0.656
# PassiveAggressiveRegressor 의 정답률:  0.7458039128719007 0.5491
# PoissonRegressor 의 정답률:  0.6749600710148136 0.5982
# QuantileRegressor 의 정답률:  -0.020280478327197038 -0.0387
# RANSACRegressor 의 정답률:  0.5151292598082302 0.199
# RadiusNeighborsRegressor 의 정답률:  0.41191760158788593 0.3148
# RandomForestRegressor 의 정답률:  0.9265962166515593 0.8296
# Ridge 의 정답률:  0.8087497007195746 0.6835
# RidgeCV 의 정답률:  0.8116598578372443 0.6833
# SGDRegressor 의 정답률:  0.8274882347568243 0.6664
# SVR 의 정답률:  0.6597910766772523 0.5398
# StackingRegressor 은 에러
# TheilSenRegressor 의 정답률:  0.7805457993164816 0.6529
# TransformedTargetRegressor 의 정답률:  0.8111288663608667 0.6822
# TweedieRegressor 의 정답률:  0.19473445117356525 0.1597
# VotingRegressor 은 에러
# PS D:\Study>  d:; cd 'd:\Study'; & 'C:\ProgramData\Anaconda3\envs\tf270gpu\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '51239' '--' 'd:\Study\ml\m06_kfold_all)estimator_6_diabets.py' 
# ARDRegression 의 정답률:  0.498748289056254 0.4906
# AdaBoostRegressor 의 정답률:  0.34846126052749715 0.467
# BaggingRegressor 의 정답률:  0.24922023412486982 0.467
# BayesianRidge 의 정답률:  0.5014366863847451 0.4869
# CCA 의 정답률:  0.48696409064967594 0.4213
# DecisionTreeRegressor 의 정답률:  -0.21251647372830607 0.0194
# DummyRegressor 의 정답률:  -0.00015425885559339214 -0.0051
# ElasticNet 의 정답률:  0.11987522766332959 0.1261
# ElasticNetCV 의 정답률:  0.48941369735908524 0.4826
# ExtraTreeRegressor 의 정답률:  -0.11374802290944874 -0.0254
# ExtraTreesRegressor 의 정답률:  0.3916612968702492 0.4856
# GammaRegressor 의 정답률:  0.07219655012236648 0.0688
# GaussianProcessRegressor 의 정답률:  -7.547010959418039 -6.5316
# GradientBoostingRegressor 의 정답률:  0.3862168253578906 0.447
# HistGradientBoostingRegressor 의 정답률:  0.28899497703380905 0.4346
# HuberRegressor 의 정답률:  0.5068530513878713 0.4829
# IsotonicRegression 은 에러
# KNeighborsRegressor 의 정답률:  0.3741821819765594 0.4042
# KernelRidge 의 정답률:  0.4802268722469346 0.4868
# Lars 의 정답률:  0.4919866521464151 0.0658
# LarsCV 의 정답률:  0.5010892359535754 0.4585
# Lasso 의 정답률:  0.46430753276688697 0.473
# LassoCV 의 정답률:  0.4992382182931273 0.4883
# LassoLars 의 정답률:  0.3654388741895792 0.3999
# LassoLarsCV 의 정답률:  0.4951942790678243 0.4829
# LassoLarsIC 의 정답률:  0.49940515175310685 0.4834
# LinearRegression 의 정답률:  0.5063891053505036 0.4838
# LinearSVR 의 정답률:  0.14945390399691316 0.1867
# MLPRegressor 의 정답률:  -0.4868977699289174 -0.5992
# MultiOutputRegressor 은 에러
# MultiTaskElasticNet 은 에러
# MultiTaskElasticNetCV 은 에러
# MultiTaskLasso 은 에러
# MultiTaskLassoCV 은 에러
# NuSVR 의 정답률:  0.12527149380257419 0.118
# OrthogonalMatchingPursuit 의 정답률:  0.3293449115305741 0.3127
# OrthogonalMatchingPursuitCV 의 정답률:  0.44354253337919725 0.4755
# PLSCanonical 의 정답률:  -0.9750792277922931 -1.2659
# PLSRegression 의 정답률:  0.4766139460349792 0.4836
# PassiveAggressiveRegressor 의 정답률:  0.4766378542920334 0.4894
# PoissonRegressor 의 정답률:  0.4823231874912104 0.4722
# QuantileRegressor 의 정답률:  -0.02193924207064546 -0.0254
# RANSACRegressor 의 정답률:  0.08488867991760607 0.0876
# RadiusNeighborsRegressor 의 정답률:  0.14407236562185122 0.1338
# RandomForestRegressor 의 정답률:  0.36094061623172125 0.476
# Ridge 의 정답률:  0.49950383964954104 0.4869
# RidgeCV 의 정답률:  0.49950383964954104 0.4851
# SGDRegressor 의 정답률:  0.49707739201119283 0.4863
# SVR 의 정답률:  0.12343791188320263 0.1196
# StackingRegressor 은 에러
# TheilSenRegressor 의 정답률:  0.5036362888903732 0.4786
# TransformedTargetRegressor 의 정답률:  0.5063891053505036 0.4838
# TweedieRegressor 의 정답률:  0.07335459385974419 0.0765
# VotingRegressor 은 에러
# PS D:\Study>  d:; cd 'd:\Study'; & 'C:\ProgramData\Anaconda3\envs\tf270gpu\python.exe' 'c:\Users\bitcamp\.vscode\extensions\ms-python.python-2021.12.1559732655\pythonFiles\lib\python\debugpy\launcher' '51248' '--' 'd:\Study\ml\m06_kfold_all)estimator_7_kaggle_bike.py' 
# ARDRegression 의 정답률:  0.24926480107309645 0.2606
# AdaBoostRegressor 의 정답률:  0.219762847805259 0.2099
# BaggingRegressor 의 정답률:  0.21848315434614507 0.2467
# BayesianRidge 의 정답률:  0.24957561720517973 0.2604
# CCA 의 정답률:  -0.1875185306363376 -0.12
# DecisionTreeRegressor 의 정답률:  -0.21697291892420312 -0.1919
# DummyRegressor 의 정답률:  -0.0006494197429334214 -0.0011
# ElasticNet 의 정답률:  0.060591067855856995 0.0609
# ElasticNetCV 의 정답률:  0.24125986412862332 0.2494
# ExtraTreeRegressor 의 정답률:  -0.1953083038037977 -0.1756
# ExtraTreesRegressor 의 정답률:  0.14738882651396645 0.1956
# GammaRegressor 의 정답률:  0.036102875217516206 0.023