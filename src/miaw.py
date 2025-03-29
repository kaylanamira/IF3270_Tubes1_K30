# NumPy-based FFNN with enhancements
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from tqdm import tqdm

def relu(x):
    return np.maximum(0, x)

def d_relu(x):
    return (x > 0).astype(float)

def sigmoid(x):
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def d_sigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def linear(x):
    return x

def d_linear(x):
    return np.ones_like(x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def d_softmax_crossentropy(y_pred, y_true):
    return (y_pred - y_true) / y_true.shape[0]

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def d_leaky_relu(x):
    return np.where(x > 0, 1, 0.01)

def swish(x):
    s = sigmoid(x)
    sw = s * x
    return sw

def d_swish(x):
    s = sigmoid(x)
    return s * (1 + x * (1 - s))

def get_activation(name):
    return {
        'relu': (relu, d_relu),
        'sigmoid': (sigmoid, d_sigmoid),
        'tanh': (tanh, d_tanh),
        'linear': (linear, d_linear),
        'softmax': (softmax, None),
        'leaky_relu': (leaky_relu, d_leaky_relu),
        'swish': (swish, d_swish),
    }[name]

# Loss functions
def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def d_mse(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_true.shape[0]

def bce(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def d_bce(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])

def cce(y_pred, y_true):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

class Layer:
    def __init__(self, n_in, n_out, activation='linear', weight_init='xavier', use_rmsnorm=False):
        self.activation_name = activation
        self.act_fn, self.d_act_fn = get_activation(activation)
        self.use_rmsnorm = use_rmsnorm

        if weight_init == 'xavier':
            limit = np.sqrt(6 / (n_in + n_out))
            self.W = np.random.uniform(-limit, limit, (n_out, n_in))
        elif weight_init == 'he':
            std = np.sqrt(2 / n_in)
            self.W = np.random.randn(n_out, n_in) * std
        elif weight_init == 'uniform':
            self.W = np.random.uniform(-1, 1, (n_out, n_in))
        elif weight_init == 'normal':
            self.W = np.random.randn(n_out, n_in)
        elif weight_init == 'zero':
            self.W = np.zeros((n_out, n_in))
        else:
            raise ValueError(f"Unknown weight_init: {weight_init}")

        self.b = np.zeros((n_out,))
        self.cache = {}

    def rmsnorm(self, x, epsilon=1e-5):
        mean_square = np.mean(x ** 2, axis=1, keepdims=True)
        norm = x / np.sqrt(mean_square + epsilon)
        return norm

    def forward(self, x):
        z = x @ self.W.T + self.b
        if self.use_rmsnorm:
            z = self.rmsnorm(z)
        a = self.act_fn(z) if self.activation_name != 'softmax' else softmax(z)
        self.cache = {'x': x, 'z': z, 'a': a}
        return a

    def backward(self, grad_output, y_true=None):
        x = self.cache['x']
        z = self.cache['z']
        a = self.cache['a']

        if self.activation_name == 'softmax':
            dz = d_softmax_crossentropy(a, y_true)
        else:
            dz = grad_output * self.d_act_fn(z)

        dW = dz.T @ x / x.shape[0]
        db = np.mean(dz, axis=0)
        dx = dz @ self.W
        return dx, dW, db

class FFNN:
    def __init__(self, layer_sizes, activations, loss_function='mse', weight_init='xavier', use_rmsnorm=False, reg_type=None):
        self.layer_sizes = layer_sizes
        self.activations = activations
        self.use_rmsnorm = use_rmsnorm
        self.weight_init = weight_init
        self.loss_function = loss_function
        self.learning_parameters = {
            "loss_function" : self.loss_function,
            "weight_init" : weight_init,
            "use_rmsnorm" : use_rmsnorm,
            "reg_type" : reg_type
        }

        self.layers = []
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i+1], activations[i], weight_init, use_rmsnorm))

        if loss_function == 'mse':
            self.loss_fn = mse
            self.d_loss_fn = d_mse
        elif loss_function == 'bce':
            self.loss_fn = bce
            self.d_loss_fn = d_bce
        elif loss_function == 'cce':
            self.loss_fn = cce
            self.d_loss_fn = None
        else:
            raise ValueError(f"Unsupported loss function: {loss_function}")

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, y_pred, y_true):
        grads = []
        grad = self.d_loss_fn(y_pred, y_true) if self.d_loss_fn else None

        for i in reversed(range(len(self.layers))):
            if self.layers[i].activation_name == 'softmax':
                grad, dW, db = self.layers[i].backward(grad_output=None, y_true=y_true)
            else:
                grad, dW, db = self.layers[i].backward(grad)
            grads.append((dW, db))
        return grads[::-1]

    def update_weights(self, grads, lr, reg_type=None, lambda_reg=0.0):
        for i, (layer, (dW, db)) in enumerate(zip(self.layers, grads)):
            if reg_type == 'l2':
                dW += lambda_reg * layer.W
            elif reg_type == 'l1':
                dW += lambda_reg * np.sign(layer.W)
            layer.W -= lr * dW
            layer.b -= lr * db

    def fit(self, x, y, x_val, y_val, epochs=100, lr=0.01, batch_size=32, verbose=1, lambda_reg=0.0):
        reg_type = self.learning_parameters["reg_type"]
        history = {'train_loss': [], 'val_loss': []}
        for epoch in range(epochs):
            indices = np.arange(x.shape[0])
            np.random.shuffle(indices)
            x, y = x[indices], y[indices]
            batch_losses = []

            if verbose:
                pbar = tqdm(total=x.shape[0], desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

            for i in range(0, x.shape[0], batch_size):
                xb = x[i:i+batch_size]
               
                yb = y[i:i+batch_size]

                y_pred = self.forward(xb)
                loss = self.loss_fn(y_pred, yb)

                if reg_type == 'l2':
                    loss += 0.5 * lambda_reg * sum(np.sum(layer.W ** 2) for layer in self.layers)
                elif reg_type == 'l1':
                    loss += lambda_reg * sum(np.sum(np.abs(layer.W)) for layer in self.layers)

                grads = self.backward(y_pred, yb)
                if epoch == epochs - 1:
                    for i, layer in enumerate(self.layers):
                        layer.grad = grads[i]
                self.update_weights(grads, lr, reg_type, lambda_reg)

                batch_losses.append(loss)

                if verbose:
                    pbar.update(xb.shape[0])

            if verbose:
                pbar.close()

            avg_train_loss = np.mean(batch_losses)
            y_val_pred = self.forward(x_val)
            val_loss = self.loss_fn(y_val_pred, y_val)

            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            if verbose:
                print(f"Epoch {epoch+1} Completed - Train Loss: {avg_train_loss:.4f} - Val Loss: {val_loss:.4f}")

        return history

    def predict(self, x):
        y_pred = self.forward(x)
        if self.loss_function == "cce":
            return np.argmax(y_pred, axis=1)
        elif self.loss_function == "bce":
            return (y_pred >= 0.5).astype(int).flatten()
        elif self.loss_function == "mse":
            return y_pred

    def get_weights(self):
        return [(layer.W.copy(), layer.b.copy()) for layer in self.layers]

    def set_weights(self, weights):
        for layer, (W, b) in zip(self.layers, weights):
            layer.W = np.array(W)
            layer.b = np.array(b).flatten()  # Ensure bias is 1D


    def to_json(self):
        return {
            "training_config": {
                "model": {
                    "input_size": self.layer_sizes[0],
                    "layers": [
                        {
                            "number_of_neurons": self.layer_sizes[i+1],
                            "activation_function": self.activations[i],
                            "use_rmsnorm": self.use_rmsnorm
                        }
                        for i in range(len(self.activations))
                    ]
                },
                "learning_parameters": self.learning_parameters
            },
            "results": {
                "final_weights": [
                    [w.tolist() + [b.tolist()] for w, b in zip(layer.W, layer.b.reshape(-1, 1))]
                    for layer in self.layers
                ]
            }
        }

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_json(), f, indent=2)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        config = data["training_config"]
        model_info = config["model"]
        learning_parameters = config["learning_parameters"]
        loss_function = learning_parameters["loss_function"]
        weight_init = learning_parameters["weight_init"]
        use_rmsnorm = learning_parameters["use_rmsnorm"]
        reg_type = learning_parameters.get("reg_type", None)
        layers = model_info["layers"]

        layer_sizes = [model_info["input_size"]] + [l["number_of_neurons"] for l in layers]
        activations = [l["activation_function"] for l in layers]
        use_rmsnorm = layers[0].get("use_rmsnorm", False)

        model = FFNN(layer_sizes, activations, use_rmsnorm=use_rmsnorm, 
                     loss_function=loss_function, weight_init=weight_init, reg_type=reg_type)

        final_weights = data.get("results", {}).get("final_weights", [])
        if final_weights:
            weights = []
            for layer_weights in final_weights:
                W = [w[:-1] for w in layer_weights]
                B = [w[-1] for w in layer_weights]
                weights.append((np.array(W), np.array(B).flatten())) 
            model.set_weights(weights)


        return model
    
    def plot_weight_distribution(self, layer_indices):
        plt.figure(figsize=(5, 3))
        sns.set_style("whitegrid")

        for idx in layer_indices:
            if idx >= len(self.layers):
                print(f"Layer {idx} dont exists.")
                continue

            # if idx == 0:
            #     continue

            weights = self.layers[idx].W.flatten()
            # plt.hist(weights, bins=20, alpha=0.6, label=f'Layer {idx+1}')
            sns.kdeplot(weights, label=f'Layer {idx+1}', fill=True, alpha=0.4, bw_adjust=0.5)

        plt.xlabel('Weight Value')
        plt.ylabel('Density')
        plt.title('Weight Distribution per Layer')
        plt.legend()
        plt.show()

    def plot_gradient_distribution(self, layer_indices):
        plt.figure(figsize=(5, 3))

        for idx in layer_indices:
            if idx >= len(self.layers):
                print(f"Layer {idx} dont exists.")
                continue

            gradients = np.concatenate([g.flatten() for g in self.layers[idx].grad])
            sns.kdeplot(gradients, label=f'Layer {idx+1}', fill=True, alpha=0.4, bw_adjust=0.5)

            plt.xlabel('Gradient Value')
            plt.ylabel('Density')
            plt.title('Gradient Distribution per Layer')
            plt.legend()
            plt.show()

            return gradients

    def plot_loss_history(self,history):
        plt.figure(figsize=(4, 3))
        plt.plot(history["train_loss"], label="Training Loss", marker='o')
        plt.plot(history["val_loss"], label="Validation Loss", marker='s')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Over Epochs")
        plt.legend()
        plt.grid(True)
        plt.show()