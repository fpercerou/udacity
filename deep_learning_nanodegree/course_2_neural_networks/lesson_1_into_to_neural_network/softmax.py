import numpy as np

def softmax(L):
    # function that takes as input a list of numbers, and returns
    # the list of values given by the softmax function.
    numerator = np.exp(L)
    denominator = np.sum(numerator)
    return numerator / denominator
