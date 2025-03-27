# MiawLentera (our version of PyTorch)

import random
import math
import numpy as np
from value import Value
from tqdm import tqdm
import json
from utils import get_loss_function

class RMSNorm:
    def __init__(self, dim, epsilon=1e-5):
        self.epsilon = Value(epsilon)
        self.gamma = [Value(1.0) for _ in range(dim)]
    
    def __call__(self, x):
        mean_sq = sum(xi ** 2 for xi in x) / Value(len(x))
        norm_factor = (mean_sq + self.epsilon) ** 0.5
        return [(xi / norm_factor) * self.gamma[i] for i, xi in enumerate(x)]

    def parameters(self):
        return self.gamma

class Module:
    def zero_grad(self):
        """Reset gradients to zero for all parameters in the model."""
        for p in self.parameters():
            p.grad = 0  

    def parameters(self):
        """Returns a list of parameters to be updated during training."""
        return []


# class Neuron(Module):
#     """
#     A neuron with adjustable weights, bias, and activation function.

#     Parameters:
#     - n_in (int): Number of input features.
#     - activation (str): Activation function ('linear', 'relu', 'sigmoid', 'tanh', 'leaky_relu', 'swish').
#     - weight_init (str): Initialization method for weights ('zero', 'uniform', 'normal').
#     - weight_params (dict): Parameters for weight initialization (mean/variance for normal, lower/upper for uniform).
#     - seed (int, optional): Random seed for weight.
#     """

#     def __init__(self, n_in, activation='linear', weight_init='uniform', weight_params={}, seed=None):            
#         self.activation = activation.lower()
#         self.w, self.b = self.initialize_weights(n_in, weight_init, weight_params, seed)  

#     def initialize_weights(self, n_in, weight_init, weight_params, seed):
#         if seed is not None:
#             random.seed(seed)

#         if weight_init == 'zero':
#             return ([Value(0) for _ in range(n_in)], Value(0))
        
#         if weight_init == 'uniform':
#             lower = weight_params.get('lower', -1)
#             upper = weight_params.get('upper', 1)
#             return ([Value(random.uniform(lower, upper)) for _ in range(n_in)], Value(random.uniform(lower, upper)))
        
#         if weight_init == 'normal':
#             mean = weight_params.get('mean', 0)
#             variance = weight_params.get('variance', 1)
#             std_dev = math.sqrt(variance)
#             return ([Value(random.gauss(mean, std_dev)) for _ in range(n_in)], Value(random.gauss(mean, std_dev)))
        
#         # xavier initialization (uniform)
#         if weight_init == 'xavier':
#             limit = math.sqrt(1 / n_in)
#             return ([Value(random.uniform(-limit, limit)) for _ in range(n_in)], Value(random.uniform(-limit, limit)))

#         # he initialization (normal)
#         if weight_init == 'he':
#             std_dev = math.sqrt(2 / n_in)
#             return ([Value(random.gauss(0, std_dev)) for _ in range(n_in)], Value(random.gauss(0, std_dev)))
        
#     def __call__(self, x):
#         """
#         Computes the output of the neuron given an input.

#         Parameters:
#         - x (list of Value): Input values.

#         Returns:
#         - Value: Activated neuron output.
#         """
#         net = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

#         if self.activation == 'relu':
#             return net.relu()
#         if self.activation == 'sigmoid':
#             return net.sigmoid()
#         if self.activation == 'tanh':
#             return net.tanh()
#         if self.activation == 'leaky_relu':
#             return net.leaky_relu()
#         if self.activation == 'swish':
#             return net.swish()
#         if self.activation == 'softmax':
#             return net.softmax()
    
#         return net

#     def parameters(self):
#         return self.w + [self.b]

#     def __repr__(self):
#         """Returns a readable string representation of the neuron."""
#         return f"Neuron({len(self.w)}, activation={self.activation})"

#     def get_weights(self):
#         """Returns the weights and bias of the neuron."""
#         return self.w

class Neuron(Module):
    """
    A neuron with adjustable weights, bias, and activation function.

    Parameters:
    - n_in (int): Number of input features.
    - weight_init (str): Initialization method for weights ('zero', 'uniform', 'normal').
    - weight_params (dict): Parameters for weight initialization (mean/variance for normal, lower/upper for uniform).
    - seed (int, optional): Random seed for weight.
    """

    def __init__(self, n_in, weight_init='uniform', weight_params={}, seed=None):            
        self.w, self.b = self.initialize_weights(n_in, weight_init, weight_params, seed)  

    def initialize_weights(self, n_in, weight_init, weight_params, seed):
        if seed is not None:
            random.seed(seed)

        if weight_init == 'zero':
            return ([Value(0) for _ in range(n_in)], Value(0))
        
        if weight_init == 'uniform':
            lower = weight_params.get('lower', -1)
            upper = weight_params.get('upper', 1)
            return ([Value(random.uniform(lower, upper)) for _ in range(n_in)], Value(random.uniform(lower, upper)))
        
        if weight_init == 'normal':
            mean = weight_params.get('mean', 0)
            variance = weight_params.get('variance', 1)
            std_dev = math.sqrt(variance)
            return ([Value(random.gauss(mean, std_dev)) for _ in range(n_in)], Value(random.gauss(mean, std_dev)))
        
        # xavier initialization (uniform)
        if weight_init == 'xavier':
            limit = math.sqrt(1 / n_in)
            return ([Value(random.uniform(-limit, limit)) for _ in range(n_in)], Value(random.uniform(-limit, limit)))

        # he initialization (normal)
        if weight_init == 'he':
            std_dev = math.sqrt(2 / n_in)
            return ([Value(random.gauss(0, std_dev)) for _ in range(n_in)], Value(random.gauss(0, std_dev)))
        
    def __call__(self, x):
        """
        Computes the output of the neuron given an input.

        Parameters:
        - x (list of Value): Input values.

        Returns:
        - Value: Activated neuron output.
        """
        return sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        """Returns a readable string representation of the neuron."""
        return f"Neuron({len(self.w)})"

    def get_weights(self):
        """Returns the weights and bias of the neuron."""
        return self.w

    
# class Layer(Module):
#     """
#     A layer consisting of multiple neurons.

#     Parameters:
#     - n_in (int): Number of inputs to each neuron.
#     - n_neuron (int): Number of neurons in the layer.
#     - kwargs: Additional arguments passed to each neuron.
#     """
#     def __init__(self, n_in, n_neuron, **kwargs):
#         self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_neuron)]

#     def __call__(self, x):
#         """
#         Computes the output of the layer given an input.

#         Parameters:
#         - x (list of Value): Input values.

#         Returns:
#         - list of Value: Layer output (each neuron’s output).
#         """
#         out = [n(x) for n in self.neurons] #array of neuron outputs
#         return out[0] if len(out) == 1 else out

#     def parameters(self):
#         return [p for n in self.neurons for p in n.parameters()]

#     def __repr__(self):
#         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class Layer(Module):
    # def __init__(self, n_in, n_neuron, **kwargs):
    #     self.activation = kwargs.get("activation", "linear")
    #     self.use_rmsnorm = kwargs.get("use_rmsnorm", False)
    #     self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_neuron)]
    #     self.norm = RMSNorm(n_neuron) if self.use_rmsnorm else None

    def __init__(self, n_in, n_neuron, **kwargs):
        self.activation = kwargs.get("activation", "linear")
        self.use_rmsnorm = kwargs.get("use_rmsnorm", False)

        neuron_kwargs = {
            k: kwargs[k]
            for k in ['weight_init', 'weight_params', 'seed']
            if k in kwargs
        }

        self.neurons = [Neuron(n_in, **neuron_kwargs) for _ in range(n_neuron)]
        self.norm = RMSNorm(n_neuron) if self.use_rmsnorm else None


    def __call__(self, x):
        raw_out = [neuron(x) for neuron in self.neurons]

        if self.use_rmsnorm:
            raw_out = self.norm(raw_out)
        return [self.activate(o) for o in raw_out] if len(raw_out) > 1 else self.activate(raw_out[0])

    def activate(self, x):
        return {
            'relu': x.relu(),
            'sigmoid': x.sigmoid(),
            'tanh': x.tanh(),
            'swish': x.swish(),
            'leaky_relu': x.leaky_relu(),
            'linear': x
        }.get(self.activation, x)

    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        if self.use_rmsnorm:
            params += self.norm.parameters()
        return params

    def __repr__(self):
        return f"Layer(activation={self.activation}, use_rmsnorm={self.use_rmsnorm})"


    
    def softmax(self, values):
        data = [v.data for v in values]
        max_val = max(data)
        exp_shifted = [math.exp(v.data - max_val) for v in values]
        sum_exp = sum(exp_shifted)

        softmax_values = [
            Value(e / sum_exp, (v,), 'softmax') for v, e in zip(values, exp_shifted)
        ]

        def _backward():
            for i, v_i in enumerate(softmax_values):
                for j, v_j in enumerate(values):
                    delta_ij = 1 if i == j else 0
                    grad = v_i.data * (delta_ij - softmax_values[j].data) * v_i.grad
                    v_j.grad += grad

        for v in softmax_values:
            v._backward = _backward

        return softmax_values

class FFNN(Module):
    """
    Our customized ver of MLPClassifier from sklearn.

    Parameters:
    - layers (list of Layer): List of layers forming the network.
    - lr (float): Learning rate for gradient descent.
    - activations (str): Activation function to use.
    - loss_function (str): Loss function to use.
    """
    def __init__(self, layer_sizes, activations, loss_function='mse', use_rmsnorm=False, **kwargs):
        self.layers = [
            Layer(layer_sizes[i], layer_sizes[i+1], activation=activations[i], use_rmsnorm=use_rmsnorm, **kwargs)
            for i in range(len(layer_sizes) - 1)
        ]
        self.loss_fn = get_loss_function(loss_function)
        self.loss_function = loss_function
        self.learning_parameters = {
            "use_rmsnorm": use_rmsnorm
        }

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"FFNN of [{', '.join(str(layer) for layer in self.layers)}]"
  
    def forward_propagate(self, x):
        return [self(x) for x in x]
    
    def backward_propagate(self, loss):
        self.zero_grad()
        loss.backward_propagate()

    def update_weights(self, lr):
        # param in this model is weight (w) and bias (b)
        for param in self.parameters():
            param.data += -lr * param.grad

    def fit(self, x, y, epochs=100, lr=0.01, batch_size=1, verbose=1,regularization_type=None, lambda_reg=0.0):
        """
        Train the model .

        Parameters:
        - x (list of list of Value): Input values.
        - y (list of Value): Target values.
        - epochs (int): Number of training epochs.
        - lr (float): Learning rate for gradient descent.
        - batch_size (int): Number of samples per batch.
        - verbose (int):  0 (dont print) or 1 (print training progress)
        """
        history = {"train_loss": [], "val_loss": []}
        self.learning_parameters.update({
            "learning_rate": lr,
            "batch_size": batch_size,
            "epochs": epochs,
            "regularization_type": regularization_type,
            "lambda": lambda_reg
        })


        for epoch in range(epochs):
            total_loss = 0
            correct = 0

            x_batches = [x[i:i + batch_size] for i in range(0, len(x), batch_size)]
            y_batches = [y[i:i + batch_size] for i in range(0, len(y), batch_size)]
            num_batches = len(x_batches)
            
            for i in range(num_batches):
                with tqdm(
                    total=1, 
                    dynamic_ncols=True,
                    desc=f"Batch {i+1}/{num_batches}",
                    bar_format="{l_bar}{bar} {n_fmt}/{total_fmt} [{postfix}]",
                    disable=(verbose == 0)
                ) as pbar:
                    y_pred = self.forward_propagate(x_batches[i])
                    batch_loss = self.loss_fn(y_pred, y_batches[i])

                    # regularization
                    if regularization_type == 'l1':
                        reg_term = lambda_reg * sum(abs(p) for p in self.parameters())
                        batch_loss += reg_term
                    elif regularization_type == 'l2':
                        reg_term = lambda_reg * sum((p * p) for p in self.parameters())
                        batch_loss += reg_term

                    total_loss += batch_loss.data
                    
                    if self.loss_function == "cce":
                        y_pred_labels = [np.argmax([prob.data for prob in yout]) for yout in y_pred]    
                        correct += sum(yt == y_pred_label for yt, y_pred_label in zip((np.argmax(y) for y in y_batches[i]), y_pred_labels))
                    elif self.loss_function == "bce":
                        y_pred_labels = [0 if y.data < 0.5 else 1 for y in y_pred] 
                        print(y_pred_labels)
                        correct += sum(yt == y_pred_label for yt, y_pred_label in zip(y_batches[i], y_pred_labels)) 

                    self.backward_propagate(batch_loss)
                    self.update_weights(lr)
                    if verbose == 1:
                        pbar.set_postfix_str(f"Epoch-{epoch+1}, Train Loss: {batch_loss.data:.5f}")
                        pbar.update(1)

            avg_batch_loss = total_loss / num_batches
            train_accuracy = (correct / sum(len(batch) for batch in y_batches)) * 100
            history["train_loss"].append(avg_batch_loss)

            if verbose == 1:
                print(f"Epoch {epoch+1}, Avg Loss: {avg_batch_loss:.4f}, Accuracy: {train_accuracy:.2f}%\n\n")

        return history

    def predict(self, x):
        if self.loss_function == "cce":
            return [np.argmax([prob.data for prob in yout]) for yout in self.forward_propagate(x)]
        elif self.loss_function == "bce":
            return [0 if yout.data < 0.5 else 1 for yout in self.forward_propagate(x)] 

    def get_final_weights(self):
        return [
            [
                [w.data for w in neuron.w] + [neuron.b.data]
                for neuron in layer.neurons
            ] for layer in self.layers
        ]
    

    def to_json(self):
        return {
            "training_config": {
                "model": {
                    "input_size": len(self.layers[0].neurons[0].w),
                    "layers": [
                        {
                            "number_of_neurons": len(layer.neurons),
                            "activation_function": layer.activation,
                            "use_rmsnorm": layer.use_rmsnorm
                        }
                        for layer in self.layers
                    ]
                },
                "learning_parameters": self.learning_parameters
            },
            "results": {
                "final_weights": self.get_final_weights()
            }
        }

    def save(self, path):
        data = self.to_json()
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        config = data["training_config"]
        model_info = config["model"]
        layers = model_info["layers"]

        # Build FFNN config from saved structure
        layer_sizes = [model_info["input_size"]] + [l["number_of_neurons"] for l in layers]
        activations = [l["activation_function"] for l in layers]
        use_rmsnorm = config["learning_parameters"].get("use_rmsnorm", False)

        # Construct model
        model = FFNN(layer_sizes, activations, use_rmsnorm=use_rmsnorm)

        # Load saved weights
        final_weights = data.get("results", {}).get("final_weights", [])
        if final_weights:
            for layer, weights in zip(model.layers, final_weights):
                for neuron, weight_bias in zip(layer.neurons, weights):
                    *w, b = weight_bias
                    neuron.w = [Value(v) for v in w]
                    neuron.b = Value(b)

        return model




    
    # def accuracy_score(self, y_true, y_pred):
    #     y_pred_labels = [1 if yout.data > 0.5 else 0 for yout in y_pred]
    #     correct = sum(yt == yp for yt, yp in zip(y_true, y_pred_labels))
    #     return correct / len(y_true) * 100