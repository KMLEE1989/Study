import pandas as pd
import numpy as np

path = '../_data/news/news1/'
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")
sample_submission=pd.read_csv(path + "sample_submission.csv")

print(train.head())

print(train.shape)

print(sample_submission.head())

print(train.isnull().sum())

print(train['target'].value_counts())

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer #tf-idf
from sklearn.svm import LinearSVC

X = train['text']
y = train['target']

X_train,X_test, y_train,y_test =train_test_split(X,y,random_state=42,test_size=.3)

clf = Pipeline([
    ('tfidf',TfidfVectorizer()),
    ('svc',LinearSVC())
])

clf.fit(X_train,y_train)

pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(classification_report(y_test,pred))

accuracy_score(y_test, pred)

pred = clf.predict(test['text'])

sample_submission['target'] = pred

sample_submission.to_csv(path + 'gookhwas1.csv', index=False)

