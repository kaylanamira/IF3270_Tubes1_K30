import math

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)
    
    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')

        def _backward():
            self.grad += (n * (self.data ** (n - 1))) * out.grad
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    
    def backward_propagate(self):

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'relu')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        s = 1 / (1 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')

        def _backward():
            self.grad += s * (1 - s) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward

        return out
    
    def leaky_relu(self):
        out = Value(0.01 * self.data if self.data < 0 else self.data, (self,), 'leaky_relu')

        def _backward():
            self.grad += (1 if out.data > 0 else 0.01) * out.grad
        out._backward = _backward

        return out

    def swish(self):
        s = 1 / (1 + math.exp(-self.data))
        sw = s * self.data
        out = Value(sw, (self,), 'swish')

        def _backward():
            self.grad += ((s * (1 - s)) * self.data + s) * out.grad
        out._backward = _backward

        return out
    
    # rusak klo gede nominal angka
    # def softmax(values):
    #     exp_values = [math.exp(v.data) for v in values]
    #     sum_exp = sum(exp_values)
    #     softmax_values = [Value(exp_v / sum_exp, (v,), 'softmax') for v, exp_v in zip(values, exp_values)]

    #     def _backward():
    #         for i, v_i in enumerate(softmax_values):
    #             for j, v_j in enumerate(values):
    #                 delta_ij = 1 if i == j else 0
    #                 v_j.grad += v_i.data * (delta_ij - v_j.data) * v_i.grad

    #     for v in softmax_values:
    #         v._backward = _backward

    #     return softmax_values
    
    def softmax(values):
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
                    grad = v_i.data * (delta_ij - v_j.data) * v_i.grad
                    v_j.grad += grad

        for v in softmax_values:
            v._backward = _backward

        return softmax_values
    


x1 = Value(1)
x2 = Value(2)
x3 = Value(3)

softmax_outputs = Value.softmax([x1, x2, x3])

# Print results
for i, s in enumerate(softmax_outputs):
    print(f"Softmax[{i}]: {s}")

# Simulate loss gradient
softmax_outputs[0].grad = 1.0  # Example gradient from loss function
softmax_outputs[0].backward_propagate()

# Print gradients
print(f"x1 grad: {x1.grad}")
print(f"x2 grad: {x2.grad}")
print(f"x3 grad: {x3.grad}")

        