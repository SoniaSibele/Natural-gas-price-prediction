#Natural gas price prediction

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

dataset = pd.read_csv('daily_csv.csv')
print (dataset.head())
print (dataset.info())

#trim date in year month and day
dataset['year'] = pd.DatetimeIndex(dataset['Date']).year
dataset['month'] = pd.DatetimeIndex(dataset['Date']).month
dataset['day'] = pd.DatetimeIndex(dataset['Date']).day

print (dataset)
dataset.drop('Date', axis=1, inplace=True) #Droping Columns
print (dataset.isnull().any()) #Checking for null values
print (dataset.isnull().sum())

#Handling Missing values
dataset['Price'].fillna(dataset['Price'].mean(),inplace=True)
print (dataset.isnull().any())


plt.bar(dataset['month'],dataset['Price'],color='green')
plt.xlabel('Month')
plt.ylabel('Price')
plt.title('PRICE OF NATURAL GAS ON THE BASIS OF MONTHS OF A YEAR')
plt.legend()
#plt.show()

#sns.lineplot(x='month',y='Price',data=dataset,color='red')
x=dataset.iloc[:,1:4].values #inputs
y=dataset.iloc[:,0:1].values#output price only

print(x) 
print(y) 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print ("y_train:", x_train.shape)
print ("y_train:", y_train.shape)

dtr=DecisionTreeRegressor()
#fitting the model or training the model
dtr.fit(x_train,y_train)

y_pred=dtr.predict(x_test)
print (y_pred)
from sklearn.metrics import r2_score
accuracy=r2_score(y_test,y_pred)
print("Acuracia:", accuracy)

y_p=dtr.predict([[1997,1,7]])
print ("The price of the date 1997-01-07 is:", y_p)


