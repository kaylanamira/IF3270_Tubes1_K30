from src.miawlentera import FFNN, Layer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate a linearly separable dataset
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, n_clusters_per_class=1)

print(type(X))

layer = Layer(1, 3, weight_init='normal', weight_params={'variance': 1, 'mean': 45}, seed=20)

print([(neuron.w, neuron.b) for neuron in layer.neurons])

lr = 0.0001
model = FFNN([
    Layer(1, 3, activation=''),
    Layer(3, 3, activation=''),
    Layer(3, 1, activation='leaky_relu'),
])
n_epoch = 500

print([neuron for layer in model.layers for neuron in layer.neurons])

for k in range(n_epoch):
    # Forward pass
    y_pred = [model(x) for x in X]
    loss = sum((yout - yt) ** 2 for yt, yout in zip(y, y_pred))

    # Calculate accuracy
    y_pred_labels = [1 if yout.data > 0.5 else 0 for yout in y_pred]
    accuracy = sum(yt == y_pred_label for yt, y_pred_label in zip(y, y_pred_labels)) / len(y)

    # Backward pass
    model.zero_grad()
    loss.backward_propagate()

    for p in model.parameters():
        p.data += -lr * p.grad

    print(f"Epoch {k + 1}, Loss: {loss.data}, Accuracy: {accuracy * 100:.2f}%")
