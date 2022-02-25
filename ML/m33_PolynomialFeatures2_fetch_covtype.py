from lightgbm import Dataset
from sklearn.datasets import load_boston, fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


datasets = fetch_covtype()   #(506,3)(506,)
#datasets = fetch_california_housing()

x = datasets.data
y = datasets.target

print(datasets.feature_names)
print(datasets.DESCR)

    # :Attribute Information:
    #     - MedInc        median income in block group
    #     - HouseAge      median house age in block group
    #     - AveRooms      average number of rooms per household
    #     - AveBedrms     average number of bedrooms per household
    #     - Population    block group population
    #     - AveOccup      average number of household members
    #     - Latitude      block group latitude
    #     - Longitude     block group longitude

print(x.shape, y.shape)
# (20640, 8) (20640,)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), XGBRegressor())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
# [0.83841344 0.81105121 0.65897081 0.63406181 0.71122933 0.51831124
#  0.73634677]

# import sklearn
# print(sklearn.metrics.SCORERS.keys())

#######################################################PolynomialFeatures 후 ##############################################################

from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2)
xp = pf.fit_transform(x)
print(xp.shape)  #(20640, 45)

x_train, x_test, y_train, y_test = train_test_split(xp,y, test_size=0.1, random_state=66)

# model = LinearRegression()
model = make_pipeline(StandardScaler(), XGBRegressor())

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, x_train, y_train, cv=7, scoring='r2')
print(scores)
