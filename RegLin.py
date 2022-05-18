from pydataset import data 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt




pima = data('Pima.tr')
pima.plot(kind='scatter', x='skin', y='bmi')


#Train values
X_train, X_test, y_train, y_test = train_test_split(pima.skin, pima.bmi)
LR = LinearRegression()
LR.fit(X_train.values.reshape(-1,1), y_train.values)

prediction = LR.predict(X_test.values.reshape(-1,1))
plt.plot(X_test,prediction, label='Linear Regression', color = 'b')
plt.scatter(X_test,y_test, label='Actual Test Data', color = 'g', alpha=.7)
plt.show()