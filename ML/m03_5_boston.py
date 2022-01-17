from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_boston
from sklearn.metrics import accuracy_score, r2_score

datasets = load_boston()

x=datasets.data
y=datasets.target

#print(x.shape, y.shape)  #(150, 4) (150,)
#print(y)
#print(np.unique(y)) #[0 1 2]  라벨값이란?  여기서는 4래 여기서 (150,4) 그리고 (150,3) 으로 만들자! 원핫 인코딩을 이용해!
# y=to_categorical(y)
# print(y)
# print(y.shape)  #(150, 3)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,train_size=0.8, shuffle=True, random_state=66) 

# print(x_train.shape, y_train.shape)
# print(x_test.shape, y_test.shape)

from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# model1= Perceptron()
# model2=LinearSVC()
# model3=SVC()
# model4=KNeighborsClassifier()
# model5=LinearRegression()
# model6=DecisionTreeClassifier()
# model7=RandomForestClassifier()

model1=KNeighborsRegressor()
model2=LinearRegression()
model3=DecisionTreeRegressor()
model4=RandomForestRegressor()

model4.fit(x_train, y_train)

result = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model4.predict(x_test)
r2=r2_score(y_test, y_predict)
# acc = accuracy_score(y_test, y_predict)

print("RandomForestRegressor:" , result)
print("r2_score:", r2)

# perceptron X
# LinearSVC X
# SVC
# KNeighborsRegressor: 0.5900872726222293
# r2_score: 0.5900872726222293

# LinearRegression: 0.8111288663608656
# r2_score: 0.8111288663608656

# DecisionTreeRegressor: 0.7836292604913669
# r2_score: 0.7836292604913669

# RandomForestRegressor: 0.9144092592581519
# r2_score: 0.9144092592581519

