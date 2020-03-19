import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]

# divide dataset in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# create model regression linear
regression = LinearRegression()
regression.fit(X_train, y_train)

# prediction with data test
y_pred = regression.predict(X_test)

# view results with data train
plt.scatter(X_train, y_train, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue")
plt.title("Salary vs Experience year - set train")
plt.xlabel("Experience year")
plt.ylabel("Salary ($)")
plt.show()

# view results with data test
plt.scatter(X_test, y_test, c="red")
plt.plot(X_train, regression.predict(X_train), c="blue")
plt.title("Salary vs Experience year - set test")
plt.xlabel("Experience year")
plt.ylabel("Salary ($)")
plt.show()
