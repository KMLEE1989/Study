from sklearn import metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

news = fetch_20newsgroups()

x = news.data
y = news.target

cv = CountVectorizer()
x =cv.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# (7919, 130107) (7919,) (3395, 130107) (3395,)

# print(x_train)

# (0, 56979)    1
#   (0, 111322)   1
#   (0, 68532)    1
#   (0, 90379)    1
#   (0, 76032)    1
#   (0, 28615)    1
#   (0, 90774)    1
#   (0, 114579)   1
#   (0, 80638)    1
#   (0, 89860)    1
#   (0, 114455)   3
#   (0, 115475)   1
#   (0, 32311)    1
#   (0, 108252)   1
#   (0, 37565)    1
#   (0, 56283)    1
#   (0, 106030)   1
#   (0, 62224)    1
#   (0, 28601)    1
#   (0, 59860)    1
#   (0, 64186)    1
#   (0, 119737)   1
#   (0, 29463)    1
#   (0, 60150)    2
#   (0, 114625)   1
#   :     :
#   (7918, 100848)        1
#   (7918, 28827) 1
#   (7918, 84968) 1
#   (7918, 80347) 1
#   (7918, 62239) 1
#   (7918, 62481) 1
#   (7918, 103497)        1
#   (7918, 95449) 1
#   (7918, 30102) 1
#   (7918, 59852) 1
#   (7918, 62004) 1
#   (7918, 39612) 1
#   (7918, 90137) 1
#   (7918, 51116) 1
#   (7918, 54616) 1
#   (7918, 108850)        1
#   (7918, 37538) 1
#   (7918, 108649)        1
#   (7918, 93276) 2
#   (7918, 48259) 2
#   (7918, 118235)        2
#   (7918, 78773) 1
#   (7918, 27010) 1
#   (7918, 8324)  1
#   (7918, 9584)  1

from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(x_train, y_train)
pred = LR.predict(x_test)

acc = accuracy_score(pred, y_test)

# print(acc)

# 0.8621502209131076

from sklearn import svm

SVM = svm.SVC(kernel='linear')
SVM.fit(x_train, y_train)
pred = SVM.predict(x_test)
acc = accuracy_score(pred, y_test)
# print(acc)

# 0.8197349042709867

from sklearn.naive_bayes import MultinomialNB

NB = MultinomialNB()
NB.fit(x_train, y_train)
pred = NB.predict(x_test)
acc = accuracy_score(pred, y_test)
# print(acc)

# 0.81620029455081

from sklearn.feature_extraction.text import TfidfTransformer

tfidf = TfidfTransformer()
x_train_tf = tfidf.fit_transform(x_train)
x_test_tf = tfidf.fit_transform(x_test)

# NB.fit(x_train_tf, y_train)
# pred = NB.predict(x_test_tf)
# acc = accuracy_score(pred, y_test)
# print(acc)
# 0.8441826215022091

from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(x_train, y_train)
pred = DT.predict(x_test)
acc = accuracy_score(pred, y_test)
# print(acc)

# 0.6435935198821797

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimator=30, learning_rate=0.05, max_depth=3)
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
acc= accuracy_score(pred, y_test)
# print(acc)

# 0.7625920471281296

from sklearn.model_selection import cross_val_score
scores = cross_val_score(NB, x, y, cv=5)
# print(scores, scores.mean())

# [0.83870968 0.83826779 0.82368537 0.83031374 0.83642794] 0.833480903927519

from sklearn.metrics import precision_score, recall_score, f1_score

# precision = precision_score(pred, y_test, average='micro')
# recall = recall_score(pred, y_test, average='micro')
# f1 = f1_score(pred, y_test, average='micro')

# print(precision, recall, f1)
# 0.7640648011782032 0.7640648011782032 0.7640648011782032 - micro

precision = precision_score(pred, y_test, average='macro')
recall = recall_score(pred, y_test, average='macro')
f1 = f1_score(pred, y_test, average='macro')

# print(precision, recall, f1)

# 0.7702437936165573 0.7940899230482429 0.7770856973100878

from sklearn.model_selection import GridSearchCV

# GS = GridSearchCV(estimator=NB, param_grid={'alpha': [0.001, 0.002, 0.003, 0.004, 0.005]}, scoring ='accuracy', cv=10)
# GS.fit(x, y)

# print(GS.best_score_) 0.8897820965842167
# print(GS.best_params_) {'alpha': 0.001}

# print(GS.best_score_)  0.8897820965842167
# print(GS.best_params_)  {'alpha': 0.001}

from sklearn.model_selection import GridSearchCV

GS = GridSearchCV(estimator=NB, param_grid={'alpha': [0.0006, 0.0008,0.001]}, scoring ='accuracy', cv=10)
GS.fit(x, y)

# print(GS.best_score_) 0.8897820965842167
# print(GS.best_params_) {'alpha': 0.001}