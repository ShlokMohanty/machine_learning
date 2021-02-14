import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
train_data = pd.read_excel("D:/Data_Train_lyst6947.xlsx")
pd.set_option('display.max_columns',None)
train_data.head()
train_data.info()
train_data["Duration"].value_counts()
train_data.dropna(inplace = True)
train_data.isnull().sum()
train_data["journey_day"] = pd.to_datetime(train_data.Date_of_Journey, format = "%d/%m/%Y").dt.day
