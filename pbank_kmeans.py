import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv(p_data.csv)

# split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# fit KMeans model on training data
kmeans = KMeans(n_clusters=2)
kmeans.fit(train_data)

# predict cluster labels for test data
pred_labels = kmeans.predict(test_data)

# convert predicted labels to binary classification format
pred_labels = np.where(pred_labels == 0, 0, 1)

# calculate accuracy of predictions
true_labels = test_data["y"].values
acc = accuracy_score(true_labels, pred_labels)

print("Accuracy:", acc)
