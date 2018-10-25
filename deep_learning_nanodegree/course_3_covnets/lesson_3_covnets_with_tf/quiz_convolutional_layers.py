"""
Setup the strides, padding and filter weight/bias such that
the output shape is (1, 2, 2, 3).
"""
import tensorflow as tf
import numpy as np

# `tf.nn.conv2d` requires the input be 4D (batch_size, height, width, depth)
# (1, 4, 4, 1)
x = np.array([
    [0, 1, 0.5, 10],
    [2, 2.5, 1, -8],
    [4, 0, 5, 6],
    [15, 1, 2, 3]], dtype=np.float32).reshape((1, 4, 4, 1))
X = tf.constant(x)


def conv2d(input):
    # Filter (weights and bias)
    # The shape of the filter weight is (height, width, input_depth, output_depth)
    # The shape of the filter bias is (output_depth,)
    # TODO: Define the filter weights `F_W` and filter bias `F_b`.
    # NOTE: Remember to wrap them in `tf.Variable`, they are trainable parameters after all.
    filter_size_height = 3
    filter_size_width = 3
    color_channels = 1
    k_output = 3
    F_W = tf.Variable(tf.truncated_normal([filter_size_height,
                                           filter_size_width,
                                           color_channels,
                                           k_output]))
    F_b =  tf.Variable(tf.zeros(k_output))
    # TODO: Set the stride for each dimension (batch_size, height, width, depth)
    strides = [1, 1, 1, 1]
    # TODO: set the padding, either 'VALID' or 'SAME'.
    padding = 'VALID'
    # https://www.tensorflow.org/versions/r0.11/api_docs/python/nn.html#conv2d
    # `tf.nn.conv2d` does not include the bias computation so we have to add it ourselves after.
    return tf.nn.conv2d(input, F_W, strides, padding) + F_b

out = conv2d(X)



# Solution
# Here's how I did it. NOTE: there is more than one way to get the correct output shape. Your answer might differ from mine.

# def conv2d(input):
#     # create the filter (weights and bias)
#     F_W = tf.Variable(tf.truncated_normal((2, 2, 1, 3)))
#     F_b = tf.Variable(tf.zeros(3))
#     strides = [1, 2, 2, 1]
#     padding = 'VALID'
#     x = tf.nn.conv2d(input, F_W, strides, padding)
#     return tf.nn.bias_add(x, F_b)
# I want to transform the input shape (1, 4, 4, 1) to (1, 2, 2, 3). I choose 'VALID' for the padding algorithm. I find it simpler to understand, and it achieves the result I'm looking for.

# out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
# out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))
# Plugging in the values:

# out_height = ceil(float(4 - 2 + 1) / float(2)) = ceil(1.5) = 2
# out_width  = ceil(float(4 - 2 + 1) / float(2)) = ceil(1.5) = 2
# In order to change the depth from 1 to 3, I have to set the output depth of my filter appropriately:

# F_W = tf.Variable(tf.truncated_normal((2, 2, 1, 3))) # (height, width, input_depth, output_depth)
# F_b = tf.Variable(tf.zeros(3)) # (output_depth)
# The input has a depth of 1, so I set that as the input_depth of the filter.
