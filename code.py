# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 13:06:38 2022

@author: Tawfik Mohamed
"""

import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics

data = pd.read_csv("student_scores.csv")

print(data.describe())

X=data['Hours']
Y=data['Scores']
print(X.shape)

cls = linear_model.LinearRegression()
X=np.expand_dims(X, axis=1)
Y=np.expand_dims(Y, axis=1)
cls.fit(X,Y) 
prediction= cls.predict(X)
plt.scatter(X, Y)
plt.xlabel('Hours', fontsize = 20)
plt.ylabel('Scores', fontsize = 20)
plt.plot(X, prediction, color='red', linewidth = 3)
plt.show()
print('Co-efficient of linear regression',cls.coef_)
print('Intercept of linear regression model',cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y, prediction))
print('Mean Absolute Error:', metrics.mean_absolute_error(Y, prediction)) 


#Predict your GPA based on your SAT Score
Studying_Hours = float(input('Enter your Studying Hours: '))
x_test=np.array([Studying_Hours])
x_test=np.expand_dims(x_test, axis=1)
y_test=cls.predict(x_test)
print('Your predicted Score is ' + str(float(y_test[0])))

