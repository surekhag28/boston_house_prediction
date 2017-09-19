# -*- coding: utf-8 -*-
"""

Predicting Boston Housing Prices
Created on Tue Sep 19 14:57:26 2017

@author: gaikwasu
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# reading data from csv file
data = pd.read_csv("housing.csv")

# separating features and labels
features = data[data.columns.values[0:3]]
prices = data[data.columns.values[-1]]


# splitting the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(features, prices, test_size=0.20, random_state=1)

# calculates performance metric using actual and predicted price values based on the metrics chosen
def performance_metric(y_true, y_pred):
    score = r2_score(y_true, y_pred)
    return score


# performs grid search over the 'max_depth' parameter for a decision tree regressor trained on [X,y]
def fit_model(X,y):

    # creates cross validation sets from tarining data
    rs = ShuffleSplit( n_splits=10, test_size=0.10, train_size=None, random_state= 0)
    cv_sets = rs.split(X,y)
    regressor = DecisionTreeRegressor()
    params = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}
    scoring_fnc = make_scorer(performance_metric)
    grid = GridSearchCV(regressor, params, scoring_fnc, cv=cv_sets)
    grid = grid.fit(X,y)

    return grid.best_estimator_

reg = fit_model(X_train,y_train)
print("Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth']))

# predicting price for some of the unknown test data

client_data = [[5, 17, 15],
               [4, 32, 22],
               [8, 3, 12]]


for i, price in enumerate(reg.predict(client_data)):
    print("Predicted selling price for Client {}'s home : ${:,.2f}".format(i,price))




