import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv(p_data.csv)

# split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# define and fit LogisticRegression model on training data
logreg = LogisticRegression()
logreg.fit(train_data.drop("y", axis=1), train_data["y"])

# make predictions on test data
pred_labels = logreg.predict(test_data.drop("y", axis=1))

# calculate accuracy of predictions
true_labels = test_data["y"].values
acc = accuracy_score(true_labels, pred_labels)

print("Accuracy:", acc)
