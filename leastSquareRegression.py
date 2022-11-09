import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12.0, 9.0)

# Preprocessing Input data
data = pd.read_csv('C:\MSC-IT\Machine Learning\leastSquareRegression\Data.csv')
X = data.iloc[:, 0]
Y = data.iloc[:, 1]
plt.scatter(X, Y)
plt.show()

# Building the model
X_mean = np.mean(X)
Y_mean = np.mean(Y)

cov = 0
var = 0
for i in range(len(X)):
    cov += (X[i] - X_mean)*(Y[i] - Y_mean)# Calculate the covariance of X and Y
    var += (X[i] - X_mean)**2 # Calculate the variance of X
regcoeff = cov / var 
intercept = Y_mean - regcoeff*X_mean

print (regcoeff, intercept)

# Making predictions
Y_pred = regcoeff*X + intercept

plt.scatter(X, Y) # actual
plt.plot([min(X), max(X)], [min(Y_pred), max(Y_pred)], color='red') # predicted
plt.show()
