# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('glass_model_output.csv')
lasers_and_temp = dataset.iloc[:,:9]
cell_number = dataset.iloc[:,13:14]
frames = [lasers_and_temp, cell_number]
dataset = pd.concat(frames, axis = 1)

panes = pd.read_csv("pane_and_thickness.csv")
panes = panes.iloc[:,:]
print(panes.iloc[:,2:])
for i in range(panes.iloc[:,2:]):
    if i%2 == 0:
        print(panes.iloc[i:i+1,2:])

# X = dataset.iloc[:, :-1].values
# y = dataset.iloc[:, 1].values

# # Splitting the dataset into the Training set and Test set
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#
# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""
#
# # Fitting Simple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = regressor.predict(X_test)
#
# # Visualising the Training set results
# plt.scatter(X_train, y_train, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Training set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()
#
# # Visualising the Test set results
# plt.scatter(X_test, y_test, color = 'red')
# plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# plt.title('Salary vs Experience (Test set)')
# plt.xlabel('Years of Experience')
# plt.ylabel('Salary')
# plt.show()