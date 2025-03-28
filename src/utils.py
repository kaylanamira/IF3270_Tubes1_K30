from value import Value

# from miawlentera import Value
# import numpy as np



# def safe_value(x):
#     return x if isinstance(x, Value) else Value(x)



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
            total = -sum(
                yt * (yp + epsilon).log() + (Value(1) - yt) * (Value(1) - yp + epsilon).log()
                for yp, yt in zip(y_pred, y_true)
            )
            return total / n
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


# def get_loss_function(name):
#     name = name.lower()

#     if name == 'mse':
#         def mse(y_pred, y_true):
#             y_pred = np.array(y_pred)
#             y_true = np.array(y_true)
#             return sum((yp - yt) ** 2 for yp, yt in zip(y_pred, y_true)) / len(y_pred)
#         return mse

#     elif name == 'bce':
#         def binary_cross_entropy(y_pred, y_true):
#             epsilon = safe_value(1e-12)
#             return -sum(
#                 safe_value(yt) * (safe_value(yp) + epsilon).log() +
#                 (safe_value(1.0) - safe_value(yt)) * (safe_value(1.0) - safe_value(yp) + epsilon).log()
#                 for yp, yt in zip(y_pred, y_true)
#             ) / len(y_pred)


#     elif name == 'cce':
#         def categorical_cross_entropy(y_pred, y_true):
#             total = Value(0.0)
#             for yp_vec, yt_vec in zip(y_pred, y_true):
#                 total += sum(
#                     -safe_value(yt) * (safe_value(yp) + Value(1e-12)).log()
#                     for yp, yt in zip(yp_vec, yt_vec)
#                 )
#             return total / len(y_pred)


#     else:
#         raise ValueError(f"Unknown loss function: {name}")