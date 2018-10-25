import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
# np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return np.array([stepFunction(x) for x in (np.matmul(X,W)+b)])

def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # The function receives as inputs the data X, the labels y,
    # the weights W (as an array), and the bias b,
    # updates the weights and bias W, b, according to the perceptron algorithm,
    # and return W and b.
    predictions = prediction(X, W, b)
    for i, pred in enumerate(predictions):
        if pred > y[i]:
            for j, x in enumerate(X[i]):
                W[j] -= X[i][j] * learn_rate
            b -= 1 * learn_rate
        elif pred < y[i]:
            for j, x in enumerate(X[i]):
                W[j] += X[i][j] * learn_rate
            b += 1 * learn_rate
    return W, b

def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    # This function runs the perceptron algorithm repeatedly on the dataset,
    # and returns a few of the boundary lines obtained in the iterations,
    # for plotting purposes.
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines
