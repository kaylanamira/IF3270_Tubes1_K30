import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

class MNIST784Preprocessor:
    def __init__(self, normalize="minmax"):
        """
        Initialize the preprocessor.
        - normalize: "minmax" (0-1 scaling) or "standard" (mean-variance normalization)
        """
        self.normalize = normalize
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def load_and_preprocess(self, test_size=0.2, random_state=42):
        X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)

        y = y.astype(int)

        if self.normalize == "minmax":
            scaler = MinMaxScaler()
        elif self.normalize == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError("Invalid normalization method. Use 'minmax' or 'standard'.")

        X = scaler.fit_transform(X)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )

    def get_data(self):
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data has not been loaded. Call load_and_preprocess() first.")
        return self.X_train, self.X_test, self.y_train, self.y_test
