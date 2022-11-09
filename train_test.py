#Design a simple machine learning model to train the training instances and test the same

import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\0_MSc_IT_Notes\Practicals\Machine Learning\practical_01\Cars.csv")
print(df.head())

#Plot to compare Milage(Independant variable) with Sell Price(Dependant variable)
plt.scatter(df['Milage'],df['Sell Price'])
plt.title("compare Milage with Sell price")
plt.show()
#Plot to compare Age(Independant variable) with Sell Price(Dependant variable)
plt.scatter(df['Age'],df['Sell Price'])
plt.title("compare Age with Sell price")
plt.show()
X = df[['Milage','Age']] #Determine X
Y = df['Sell Price']     #Determine Y

print("\nindependent varaible x :- \n",X)
print("\ndependent variable y :- \n",Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)
print("\nlength of x_train = ",len(X_train))
print("\nlength of X_test =",len(X_test))

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
clf.fit(X_train,Y_train) # Train the model
print("\npredicted X_test :- \n",clf.predict(X_test))

print("\naccuracy = ",clf.score(X_test,Y_test)) #Calculate the accuracy

