import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Generate a more linearly separable dataset with 4 classes
X, y = make_classification(n_samples=200,  # Increased samples for better visualization
                           n_features=2,
                           n_redundant=0,
                           n_informative=2,
                           random_state=1,
                           n_clusters_per_class=1,
                           n_classes=2,
                           class_sep=4.0)  # Increase separation between classes

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 80% train, 20% test

# Plot the training and testing sets
plt.figure(figsize=(12, 6))

# Training set plot
plt.subplot(1, 2, 1)  # 1 row, 2 columns, plot 1
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title('Training Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(ticks=[0, 1, 2, 3])

# Testing set plot
plt.subplot(1, 2, 2)  # 1 row, 2 columns, plot 2
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.RdYlBu, edgecolors='k')
plt.title('Testing Set')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar(ticks=[0, 1, 2, 3])

from sklearn.preprocessing import LabelEncoder
import numpy as np

encoder = LabelEncoder() # sparse=False returns a NumPy array
y_label_encode = encoder.fit_transform(y.reshape(-1, 1))

import importlib
import miawlentera
importlib.reload(miawlentera)
from miawlentera import FFNN

model = FFNN(
    layer_sizes=[2, 1], 
    activations=['sigmoid'], 
    loss_function='bce'
)

history = model.fit(X, y_label_encode, epochs=20, batch_size=2, lr=0.01, verbose=1)