import numpy as np


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def unit_step(x):
    return np.heaviside(x, 1)


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)
