import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# import dataset and create variable dependence and independence
dataset = pd.read_csv("Data.csv")
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1:]

# replace values nan to median values for columns
# verbose = 0, replace mean of column
imputer = SimpleImputer(missing_values=np.nan, strategy="mean", verbose=0)
imputer = imputer.fit(X.iloc[:, 1:])
X.iloc[:, 1:] = imputer.transform(X.iloc[:, 1:])

# translate data categories
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])

# onehotencoder = OneHotEncoder(categories="auto")
# X.iloc[:, 0] = onehotencoder.fit_transform(X.iloc[:, 0]).toarray()

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],   # [0] columns in data
    remainder='passthrough'
)
# transform to dummy variables
X = np.array(ct.fit_transform(X), dtype=np.float)

# transform data output to dummy variables
labelencoder_y = LabelEncoder()
y.iloc[:, 0] = labelencoder_y.fit_transform(y.iloc[:, 0])

# divide dataset in test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# normal or standard variables
scaler_X = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
