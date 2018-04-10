# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('glass_and_thickness_modified.csv')
print(dataset)
X = dataset.iloc[:, :3].values
Y = dataset.iloc[:, 3:4].values

[print(f'independant: {X[i]}') for i in range(len(X))]
[print(f'dependant: {Y[i]}') for i in range(len(Y))]
#
# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 0] = labelencoder.fit_transform(X[:, 0])

onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# # Feature Scaling
# """from sklearn.preprocessing import StandardScaler
# sc_X = StandardScaler()
# X_train = sc_X.fit_transform(X_train)
# X_test = sc_X.transform(X_test)
# sc_y = StandardScaler()
# y_train = sc_y.fit_transform(y_train)"""
#
# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)

# print(X[:, -1])
for i in range(len(Y)):
    if Y[i] < 0.15:
        Y[i] = 0.191


# # Visualising the Test set results
plt.scatter(X[:, -1], Y, color = 'red')
#plt.plot(X[:, -1], regressor.predict(X), color = 'blue')
plt.title('Glass Thickness (Test set)')
plt.xlabel('Glass Thickness')
plt.ylabel('laser N DS')
plt.show()

import statsmodels.formula.api