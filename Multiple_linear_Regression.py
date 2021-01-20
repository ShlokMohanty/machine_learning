import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv("csv_file_path")
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]
dataset.head()
states = pd.get_dummies(X['column_name'],drop_first=True)
X=X.drop('column_name',axis=1)
X=pd.concat([X,states],axis=1)
X
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split( X, Y, test_size=0.2, random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
Y_pred
X_test
from sklearn.metrics import r2_score
score = r2_score(Y_test,Y_pred)
score
# the value nearest to 1 is the correct evaluated data/value.
