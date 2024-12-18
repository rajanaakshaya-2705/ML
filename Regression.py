import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Load the dataset
dataset = pd.read_csv("C:\\Users\\AKSHAYA\\Downloads\\Salary Data.csv")
# Display the first few rows of the dataset
print(dataset.head())
# Extract features (X) and target variable (y)
X = dataset.iloc[:, :-1].values  # All rows, all columns except the last
y = dataset.iloc[:, -1].values    # All rows, only the last column
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Create a linear regression model and fit it to the training data
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Output the coefficients and intercept of the model
print(f'Coefficients: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
# Make predictions on the test set
y_pred = regressor.predict(X_test)
# Visualize the training set results
plt.scatter(X_train, y_train, color='red', label='Actual Salary')
plt.plot(X_train, regressor.predict(X_train), color='blue', label='Predicted Salary')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
# Visualize the test set results
plt.scatter(X_test, y_test, color='red', label='Actual Salary')
plt.scatter(X_test, y_pred, color='blue', label='Predicted Salary')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend()
plt.show()
# Calculate and print Mean Squared Error for the test set
from sklearn.metrics import mean_squared_error
mse_test = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error (Test set): {mse_test}')
# Calculate and print Mean Squared Error for the training set
mse_train = mean_squared_error(y_train, regressor.predict(X_train))
print(f'Mean Squared Error (Training set): {mse_train}')
