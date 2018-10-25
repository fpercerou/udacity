import numpy as np
import tensorflow as tf
from six.moves import range
from datetime import datetime
import math


class OneLayerCovnet(object):
    """
    Class to implement 1-layer Convolutional neural networks.
    """
    def __init__(self, height, width, depth, nb_labels,
                 patch_height, patch_width, conv_depth,
                 stride_height=1, stride_width=1, padding='SAME'):
        super(OneLayerCovnet, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth # 1: grayscale, 3: RGB
        self.nb_labels = nb_labels
        self.patch_height = patch_height
        self.patch_width = patch_width
        self.conv_depth = conv_depth
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.padding = padding

    @property
    def required_data_shape(self):
        return (self.height, self.width, self.depth)

    @property
    def required_label_shape(self):
        return (self.nb_labels)

    def check_format(self, data, labels=None):
        """
        method to check that the data is in the correct format
        data: input data must be a 4 dimensional matrix
        (nb_obs, self.height, self.width, self.depth)
        (we do not check the nb of obs)
        each row is an image (RGB if self.depth==3, grayscale if self.depth==1)
        labels: label data must a 2 dimensional matrix (nb_obs, self.nb_labels)
        (we do not check the nb of obs)
        each row is a horizontal vector with a dummy with one and only one 1
        """
        assert data.shape[1:] == self.required_data_shape, "data.shape must be %s not %s" % (self.required_data_shape, data.shape[1:])
        if labels is not None:
            assert labels.shape[1:] == self.required_label_shape, "labels.shape must be %s not %s" % (self.required_label_shape, labels.shape[1:])
            assert all([x==1 for x in labels.sum(1)]), "labels rows must have one and only one 1 per row"

    @property
    def output_height(self):
        if self.padding=='SAME':
            return math.ceil(float(self.height) / float(self.stride_height))
        else:
            NotImplemented("padding height not implemented for '%s' padding" % self.padding)

    @property
    def output_width(self):
        if self.padding=='SAME':
            return math.ceil(float(self.width) / float(self.stride_width))
        else:
            NotImplemented("padding height not implemented for '%s' padding" % self.padding)

    def initiate_weights_and_layer(self):
        """
        initiates the weight and biases tf objects
        """
        layer_weights = tf.Variable(tf.truncated_normal([self.patch_height, self.patch_width, self.depth, self.conv_depth],
                                                       stddev=0.1),
                                  name="W")
        layer_biases = tf.Variable(tf.zeros([self.conv_depth]), name="B")
        return layer_weights, layer_biases

    def build_conv2d(self, data, layer_weights=None, layer_biases=None):
        """
        builds the final conv2d tf object
        """
        self.check_format(data)
        if layer_weights is None or layer_biases is None:
            layer_weights, layer_biases = self.initiate_weights_and_layer()
        conv = tf.nn.conv2d(data,
                            layer_weights,
                            [1, self.stride_height, self.stride_width, 1],
                            padding=self.padding)
        return tf.nn.relu(conv + layer_biases)