import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

iris= pd.read_csv('C:\MSC-IT\Machine Learning\K Nearest algorithm\iris.csv')
print(iris.head()) #Display the iris dataset
print(iris.shape) #Display the dimensions of the iris dataset
print(iris['variety'].value_counts())#Dislay the no of instances belonging to every value of the dependent variable
print(iris.columns) #Display the column_headers of the iris dataset
print(iris.values) #Display the values of the dataset

X=iris.iloc[:,:4] #Define the independent variable vector X
print(X.head())#Display X
y=iris.iloc[:,-1] #Define the dependent variable y
print(y.head())#Display y

X=preprocessing.StandardScaler().fit_transform(X)#Preprocess the data to achieve a mean of 0 and standard deviation of 1
print(X[0:4]) #Display the preprocessed data

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.3) #Create the training set and the testing set
Y_test.shape #Display the dimensions of Y_test

knnmodel=KNeighborsClassifier(n_neighbors=3) #Build the KNN model for k=3
knnmodel.fit(X_train,Y_train) #Train the KNN Model
Y_pred = knnmodel.predict(X_test) #Use the KNN model for predicting the class of the test set
print(Y_pred) #Display the result of prediction

#Calculate the accuracy of the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,Y_pred))

#Constructing the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(Y_test.values,Y_pred)
print(cm)

#Visualization of the output

cm1=pd.DataFrame(data=cm,index=['Setosa','Versicolor','Virginica'],columns=['Setosa','Versicolor','Virginica'])
print(cm1)
pred_output=pd.DataFrame(data=[Y_test.values,Y_pred],index=['Y_test','Y_pred'])
print(pred_output.transpose())