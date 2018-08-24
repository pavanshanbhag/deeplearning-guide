"""
linear_regression_demo

Overview
The goal is to predict an animal's body weight given it's brain weight. 
The model we'll be using is called [Linear Regression](http://www.statisticssolutions.com/what-is-linear-regression/). 
The dataset we're using to train our model is a list of brain weight and body weight
measurements from a bunch of animals. We'll fit our line to the data using the 
scikit learn machine learning library, then plot our graph using matplotlib.

Dependencies
* pandas
* scikit-learn
* matplotlib
"""

import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt

#read data
dataframe = pd.read_fwf(r'datasets\brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]

#train model on data
body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)

#visualize results
plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()
