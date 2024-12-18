import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv("C:\\Users\\AKSHAYA\\Downloads\\Salary Data.csv")
print(dataset.head())
X = dataset.iloc[:, :-1].values  
y = dataset.iloc[:, -1].values   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(f'Coefficients: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color='red', label='Actual Salary')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted Salary')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
plt.scatter(X_test, y_test, color='red', label='Actual Salary')
plt.scatter(X_test, y_pred, color='blue', label='Predicted Salary')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Test set): {mse_test}')
mse_train = mean_squared_error(y_train, regressor.predict(X_train))
print(f'Mean Squared Error (Training set): {mse_train}')
