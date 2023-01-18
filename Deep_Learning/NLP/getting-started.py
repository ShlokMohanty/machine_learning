import numpy as np 
import pandas as pd 
from sklearn import feature_extraction, preporcessing 
train_df = pd.read_csv("C:\Users\Shlok Mohanty\Downloads\submissions.csv")
test_df = pd.read_csv("C:\Users\Shlok Mohanty\Downloads\submissions.csv")
#quick look at our data 
train_df[train_df["target"]==0]["text"].values[1]
train_df[train_df["target"]==1]["text"].values[1]
#vector building 
#scikit consists of CountVectorizer to count the words in each tweet and turn them into the data 
#our machine learning model can process
count_vectorizer = feature_extraction.text.CountVectorizer()
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5])
## we use .todendse() here because these vectors are sparse (only non-zero elements are kept to save
#space)
print(example_train_vectors[0].todense().shape)
print(example_train_vectors[0].todense())
#(1,54)
#[[0 0 0 1 1 1 0 0 0  0 0 0 1 1 
#0 0 0 0 1  0 0 0 0 0 0 1  0 0 0 1 0 0 0 0 1 0 0 0 0 1 0 0  0 0 0 0 0 0 1 1 0 1 0]]
#the above vector tells us that it has 54 unique words (or "tokens")

