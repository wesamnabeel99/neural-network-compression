import numpy


def sigmoid(logit):
    return 1.0 / (1 + numpy.exp(-logit))
