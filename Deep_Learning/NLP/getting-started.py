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

train_vectors = count_vectorizer.fit_transform(train_df["text"])
# note that we are not using .fit_transform() here. Using just .transform()
## makes sure that the tokens in the train vectors are the only ones mapped to the test vectors 
# the train and test vectors use the same segt of tokens 
#the trai and the test vectors use the same set of tokens 
test_vectors = count_vectorizer.transform(test_df["text"])

#our model
clf= linear_model.RidgeClassifier()
#metric used here is F1 
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores
clf.fit(train_vectors, train_df["target"])
sample_submission = pd.read_csv("C:\Users\Shlok Mohanty\Downloads\submissions.csv")
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)
