import numpy as np

def cross_entropy(Y, P):
    # Function that takes as input two lists Y, P,
    # and returns the float corresponding to their cross-entropy.
    # only works for Y being binary variables and P=P(Y=1)
    return np.sum(-np.log(np.multiply(np.array(Y), np.array(P))
                  + np.multiply(1-np.array(Y), 1-np.array(P))))