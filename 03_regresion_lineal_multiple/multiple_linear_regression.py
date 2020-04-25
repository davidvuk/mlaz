import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

"""
import dataset
"""
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values


"""
convert data category to numbers
"""
label_encoder_X = LabelEncoder()
X[:, -1] = label_encoder_X.fit_transform(X[:, -1])
# convert data numbers to variable dummy
column_transform = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [-1])],  # transform columns last (-1)
    remainder='passthrough'
)
X = np.array(column_transform.fit_transform(X))
# avoid tramp variables dummy, delete first column (column dummy)
X = X[:, 1:]


"""
divide data in train and test
"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""
fit model with data train
"""
regression = LinearRegression()
regression.fit(X_train, y_train)


"""
predict model with data test
"""
y_prediction = regression.predict(X_test)


"""
optimize model using delete comeback
"""
# add ones to matrix X for simulate bias
X = np.append(np.ones((len(X), 1)).astype(int), X, axis=1)

# delete comeback, start with all features after delete feature than smaller
X_optimize = X.copy()
SL = 0.05

model_regression_OLS = sm.OLS(y, X_optimize)
result = model_regression_OLS.fit()
# regression_OLS.summary()

