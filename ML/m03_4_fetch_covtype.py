from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC,SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression  #regressor 지만 이건 분류다 명심
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_covtype

datasets = fetch_covtype()

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

model1= Perceptron()
model2=LinearSVC()
model3=SVC()
model4=KNeighborsClassifier()
model5=LogisticRegression()
model6=DecisionTreeClassifier()
model7=RandomForestClassifier()

model4.fit(x_train, y_train)

result = model4.score(x_test, y_test) 

from sklearn.metrics import accuracy_score
y_predict=model4.predict(x_test)
acc = accuracy_score(y_test, y_predict)

print("KNeighborsClassifier : ", result)

#Perceptron :  0.49991824651687133
#LinearSVC :  0.6258530330542241
#SVC :  
#KNeighborsClassifier:  
#LogisticRegression :  0.6198721203411272
#DecisionTreeClassifier :  0.940285534796864
#RandomForestClassifier :  0.9556982177740678