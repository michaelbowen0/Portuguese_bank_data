import pandas as pd
from sklearn.model_selection import train_test_split

# load dataset
data = p_data

# split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# output shape of training and testing sets
print("Training data shape:", train_data.shape)
print("Testing data shape:", test_data.shape)