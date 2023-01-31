import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# load dataset
data = pd.read_csv(p_data.csv)

# split data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# define SVC model
svc = SVC()

# define parameters to search over in grid search
param_grid = {
    "C": [0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"]
}

# perform grid search over parameters
grid_search = GridSearchCV(svc, param_grid, cv=5)
grid_search.fit(train_data.drop("y", axis=1), train_data["y"])

# get best parameters and fit SVC model with best parameters
best_params = grid_search.best_params_
best_svc = SVC(C=best_params["C"], kernel=best_params["kernel"])
best_svc.fit(train_data.drop("y", axis=1), train_data["y"])

# make predictions on test data
pred_labels = best_svc.predict(test_data.drop("y", axis=1))

# calculate accuracy of predictions
true_labels = test_data["y"].values
acc = accuracy_score(true_labels, pred_labels)

print("Accuracy:", acc)
print("Best parameters:", best_params)
