import numpy as np
import tensorflow as tf
from six.moves import range
from datetime import datetime


class OneLayerFullyConnected(object):
    """
    Class to implement 1-layer Fully Connected neural networks.
    """
    def __init__(self, input_dim, nb_hidden, nb_labels):
        super(OneLayerFullyConnected, self).__init__()
        self.nb_labels = nb_labels
        self.input_dim = input_dim
        self.nb_hidden = nb_hidden

    @property
    def required_data_shape(self):
            return (self.input_dim)

    @property
    def required_label_shape(self):
        return (self.nb_labels)

    def check_format(self, data, labels=None):
        """
        method to check that the data is in the correct format
        data: input data must be a 2 dimensional matrix
        (nb_obs, self.input_dim) (we do not check the nb of obs)
        each row is a horizontal vector
        labels: label data must a 3 dimensional matrix (nb_obs, self.nb_labels)
        (we do not check the nb of obs)
        each row is a horizontal vector with a dummy with one and only one 1
        """
        assert data.shape[1:] == self.required_data_shape, "data.shape must be %s not %s" % (self.required_data_shape, data.shape[1:])
        if labels is not None:
            assert labels.shape[1:] == self.required_label_shape, "labels.shape must be %s not %s" % (self.required_label_shape, labels.shape[1:])
            assert all([x==1 for x in labels.sum(1)]), "labels rows must have one and only one 1 per row"

    def initiate_weights_and_layer(self):
        """
        initiates the weight and biases tf objects
        """
        layer_weights = tf.Variable(tf.truncated_normal([self.input_dim, self.nb_hidden],
                                                        stddev=0.1),
                                    name="W")
        layer_biases = tf.Variable(tf.constant(1.0, shape=[self.nb_hidden]))
        return layer_weights, layer_biases

    def build_layer(self, data, layer_weights=None, layer_biases=None):
        """
        builds the final conv2d object
        """
        self.check_format(data)
        if layer_weights is None or layer_biases is None:
            layer_weights, layer_biases = self.initiate_weights_and_layer()
        return tf.nn.relu(tf.matmul(data, layer_weights) + layer_biases)

