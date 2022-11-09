#A simple machine learning model to train the training instances and test the same

print("Varsha Bamane 01")
#
#
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"D:\ML\1_Training&Testing\Cars.csv") #Read data
df.head()

plt.scatter(df['Milage'],df['Sell Price']) #Plot to compare Milage(Independant variable) with Sell Price(Dependant variable)
plt.show()
plt.scatter(df['Age'],df['Sell Price'])#Plot to compare Age(Independant variable) with Sell Price(Dependant variable)
plt.show()

X = df[['Milage','Age']] #Determine X
Y = df['Sell Price']    # Determine Y
print(X) #Display X
print(Y) #Display Y

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)#Create a training set and testing set
len(X_train)
len(X_test)

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train) # Train the model
clf.predict(X_test) # Use the trained model to predict the testing set
print(clf.score(X_test,Y_test)) #Calculate the accuracy


