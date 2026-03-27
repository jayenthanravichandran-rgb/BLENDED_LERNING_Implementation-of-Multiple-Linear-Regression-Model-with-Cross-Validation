# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
/*
Program to implement the multiple linear regression model for predicting car prices with cross-validation.
Developed by: R JAYENTHAN
RegisterNumber:  25011312   //    212225240057
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
import matplotlib.pyplot as plt

df = pd.read_csv("caras.csv")
print(df.head())

data=df.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)

X=data.drop('price',axis=1)
y=data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train,y_train)


print("Name:R JAYENTHAN ")
print("Reg No: 212225240057 ")
print("\n==== Cross Validation ====")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R-Square scores:",[f"{score:4f}" for score in cv_scores])
print(f"Average R-square:{cv_scores.mean():.4f}")

y_pred = model.predict(X_test)
print("\n=== Test Set Performance ===")
print(f"MSE: {mean_squared_error(y_test,y_pred):.2f}")
print(f"R-square:{r2_score(y_test,y_pred):.4f}")

plt.figure(figsize=(8,6))
plt.scatter(y_test,y_pred,alpha=0.6)
plt.plot([y.min(),y.max()],[y.min(),y.max()],'r--')
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Price ")
plt.ylabel("Predicted Price ")
plt.grid(True)
plt.show()

*/
```

## Output:
<img width="985" height="549" alt="image" src="https://github.com/user-attachments/assets/dcd51d80-cc29-4753-ab0f-c1b1e6c87f09" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
