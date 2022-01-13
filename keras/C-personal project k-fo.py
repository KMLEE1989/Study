from sklearn.model_selection import cross_validate, KFold
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


path = "../_data/개인프로젝트/CSV/"
dataset=pd.read_csv(path+'통합.csv',thousands=',')
num_folds = 2


#print(dataset.info()) 

# data = dataset.drop(['DATE', '지점'], axis=1)
# target = dataset['AVG TEMP(℃)']

x = dataset.drop(['DATE', '지점'],axis=1)
y = dataset['AVG TEMP(℃)'] 

x_train,x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x.shape)
print(y.shape)

kfold = KFold(n_splits=num_folds, shuffle=True)

for train, test in kfold.split(x, y):
    
  # Define the model architecture
    model = Sequential()
    model.add(Dense(64, input_dim=15)) 
    model.add(Dense(128, activation='relu'))      
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32))
    model.add(Dense(16))
    model.add(Dense(1))

model.summary()

model.compile(loss='mae',
                optimizer='adam',
                metrics=['accuracy'])

history = model.fit(x_train, y_train,
              batch_size=2,
              epochs=100,
              )

scores = model.evaluate(x_test, y_test, verbose=0)

print(scores)

[0.09212144464254379, 0.0139860138297081]
