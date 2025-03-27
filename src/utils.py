from value import Value

def get_loss_function(name):
    name = name.lower()

    if name == 'mse':
        def mse(y_pred, y_true):
            n = len(y_pred)
            return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / n
        return mse

    elif name == 'bce':
        def binary_cross_entropy(y_pred, y_true):
            n = len(y_pred)
            epsilon = Value(1e-12)
            total = sum(
                yt * (yp + epsilon).log() + (Value(1) - yt) * (Value(1) - yp + epsilon).log()
                for yp, yt in zip(y_pred, y_true)
            )
            return -total / n
        return binary_cross_entropy

    elif name == 'cce':
        def categorical_cross_entropy(y_pred, y_true):
            n = len(y_pred)
            total = Value(0.0)
            for y_vec, y_true_vec in zip(y_pred, y_true):
                total += sum(-yt * (yp + Value(1e-12)).log() for yp, yt in zip(y_vec, y_true_vec))
            return total / n
        return categorical_cross_entropy

    else:
        raise ValueError(f"Unknown loss function: {name}")
