#data preprocessing

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import dataset
dataset = pd.read_csv("C:\\Users\\OEM\\Source\\Repos\\veriphi-python\\tutorials\\Machine Learning\\data preprocessing\\Data.csv")
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values
print(f"These are our dependant variables:\n {X}")
print(f"This is a column of our independant variable:\n {Y}")

#take care of missing data by taking mean of the column
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(f"These are our dependant variables:\n {X}")

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])
one_hot_encoder = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder.fit_transform(X).toarray()

label_encoder_Y = LabelEncoder()
Y = label_encoder_Y.fit_transform(Y)

print(f"These are our dependant v after transforming:\n {X}")
print(f"These are our independant v after transforming:\n {Y}")

# Splitting dataset into  training and test set
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0)
print(f'x train: {X_train}')
print(f'x test: {X_test}')
print(f'y train: {Y_train}')
print(f'y test: {Y_test}')

# Feature scaling
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print(f'x train: {X_train}')
print(f'x test: {X_test}')
print(f'y train: {Y_train}')
print(f'y test: {Y_test}')

