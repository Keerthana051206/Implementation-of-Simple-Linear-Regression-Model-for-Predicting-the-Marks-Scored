# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the standard Libraries. 
2.Set variables for assigning dataset values. 
3.Import linear regression from sklearn. 
4.Assign the points for representing in the graph. 
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:

```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Keerthana C
RegisterNumber: 212224220047
*/
```
```
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
dataset=pd.read_csv('student_scores.csv')
print(dataset.tail())
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)

plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
```
## Output:
<img width="278" height="317" alt="image" src="https://github.com/user-attachments/assets/d9c7456d-ceb2-4069-86f1-f0fd8b2ab567" />




<img width="276" height="32" alt="image" src="https://github.com/user-attachments/assets/99fc805e-ab2c-45f2-a925-f1ea899eef97" />




<img width="352" height="259" alt="image" src="https://github.com/user-attachments/assets/fd274577-3f02-4388-9223-8fdb82410759" />




<img width="356" height="271" alt="image" src="https://github.com/user-attachments/assets/4ba0a26e-f51b-4af8-be16-17a5e08ffd5f" />




<img width="226" height="35" alt="image" src="https://github.com/user-attachments/assets/cb9dffe7-c7fc-40ed-a397-288fdcc6789e" />


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
