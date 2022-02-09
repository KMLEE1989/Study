from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
 
# Function to create model, required for KerasClassifier
def create_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=8, activation='relu'))
	model.add(Dense(4, activation='relu'))
	model.add(Dense(1, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
 
# load pima indians dataset
path = "../_data/개인프로젝트/CSV/"

dataset =pd.read_csv("pima-indians-diabetes.data.csv", delimiter=",")
 
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
 
# create model
model = KerasClassifier(build_fn=create_model, epochs=150, batch_size=10, verbose=0)
 
kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)

