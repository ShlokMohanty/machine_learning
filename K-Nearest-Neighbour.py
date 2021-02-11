
import pandas as pd
import numpy as np
zoo = pd.read_csv('D:/zoo_lyst7412.csv')
zoo.head()
zoo['type'].unique()
zoo1=zoo.drop(['animal name'],axis = 1)

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo1,test_size = 0.2)
from sklearn.model_selection import train_test_split
x_train = train.iloc[: ,0:16]
Y_train = train.iloc[: ,16]
from sklearn.neighbors import KNeighborsClassifier as KNC
acc = []
for i in range (3,50,2)
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(tarin.iloc[: ,0:16],train.iloc[: ,16])
    train_acc = np.mean(neigh.predict(train.iloc[: ,0:16])==train.iloc[: ,16])
    test_acc = np.mean(neigh.predict(test.iloc[: ,0:16])==test.iloc[: ,16])
    acc.append([train_acc,test_acc])
    import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
plt.legend(["train","test"])
plt.show()
acc1=[]
for i in range(1,12,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[: ,0:16],train.iloc[: ,16])
    train_acc1 = np.mean(neigh.predict(train.iloc[: ,0:16])==train.iloc[: ,16])
    test_acc1 = np.mean(neigh.predict(test.iloc[: ,0:16])==test.iloc[: ,16])
    acc1.append([train_acc1,test_acc1])
    import matplotlib.pyplot as plt
plt.plot(np.arange(1,12,2),[i[0] for i in acc1],"ro-")
plt.plot(np.arange(1,12,2),[i[1] for i in acc1],"bo-")
plt.legend(["train","test"])
plt.show()
neigh = KNC(n_neighbors=5)
neigh.fit(train.iloc[: ,0:16],train.iloc[: ,16])
neigh.fit(test.iloc[: ,0:16],test.iloc[: ,16])
x_train = train.iloc[: ,0:16]
y_train = train.iloc[: ,16]
Y_pred = neigh.predict(x_train)
train_acc = np.mean(y_pred==y_train)
