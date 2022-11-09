import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv("C:\MSC-IT\Machine Learning\logistic Regression\insurance_data.csv")
df.head()

plt.scatter(df.Age,df.have_insurance,marker='+',color='red')
plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(df[['Age']],df.have_insurance,test_size=0.1)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, Y_train)#Training the model
model.predict(X_test)#Prediction on the test set
model.score(X_test,Y_test)#Calculating the accuracy
model.predict_proba(X_test)#Predicting the probabilities
