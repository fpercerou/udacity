from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import pdb
import os
import numpy as np

n_input = 784  # MNIST data input (img shape: 28*28)
n_classes = 10  # MNIST total classes (0-9 digits)

# Import MNIST data
folder = os.path.dirname(os.path.abspath(__file__))
mnist = input_data.read_data_sets(os.path.join(folder, 'mnist'), one_hot=True)

# The features are already scaled and the data is shuffled
train_features = mnist.train.images
test_features = mnist.test.images

train_labels = mnist.train.labels.astype(np.float32)
test_labels = mnist.test.labels.astype(np.float32)

# Weights & bias
weights = tf.Variable(tf.random_normal([n_input, n_classes]))
bias = tf.Variable(tf.random_normal([n_classes]))

print("Memory usage:")
print("\ttrain_features:", train_features.shape[0] * train_features.shape[1] * (32 / 8), 'bytes')
print("\train_labels:", train_labels.shape[0] * train_labels.shape[1] * (32 / 8), 'bytes')
print("\tweights:", weights.shape[0] * weights.shape[1] * (32 / 8), 'bytes')
print("\tbias:", bias.shape[0] * (32 / 8), 'bytes')


# train_features Shape: (55000, 784) Type: float32
# train_labels Shape: (55000, 10) Type: float32
# weights Shape: (784, 10) Type: float32
# bias Shape: (10,) Type: float32


print("nb batches: ", int(50000 / 128) + 1)
print("last batches: ", 50000 % 128)