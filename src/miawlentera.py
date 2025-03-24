# MiawLentera (our version of PyTorch)

import random
import math
from value import Value

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, n_in, activation='linear', weight_init='uniform', weight_params={}, seed=None):            
        self.activation = activation.lower()
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
        
    def __call__(self, x):
        net = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)

        if self.activation == 'relu':
            return net.relu()
        if self.activation == 'sigmoid':
            return net.sigmoid()
        if self.activation == 'tanh':
            return net.tanh()
        if self.activation == 'leaky_relu':
            return net.leaky_relu()
        if self.activation == 'swish':
            return net.swish()
    
        return net

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron({len(self.w)}, activation={self.activation})"

class Layer(Module):
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class FFNN(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"FFNN of [{', '.join(str(layer) for layer in self.layers)}]"