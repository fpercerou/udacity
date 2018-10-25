import tensorflow as tf
import numpy as np

class Dropout(object):
    """
    Class to implement a 1-layer Dropout.
    """
    def __init__(self, height, width, depth, nb_labels,
                 keep_prob, **tf_dropout_kwargs):
        super(Dropout, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth # 1: grayscale, 3: RGB
        self.nb_labels = nb_labels
        self.keep_prob = keep_prob
        self.tf_dropout_kwargs = tf_dropout_kwargs
        warning_log = "To do: remove Dropout for validation and test!"
        print(warning_log)

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
        return self.height


    @property
    def output_width(self):
        return self.width

    @property
    def output_depth(self):
        return self.depth

    def initiate_weights_and_layer(self):
        """
        initiates the weight and biases tf objects
        """
        return None, None


    def build_dropout(self, data):
        """
        builds the final pooling tf object
        """
        self.check_format(data)
        return tf.nn.dropout(data,
                             keep_prob=self.keep_prob,
                             **(self.tf_dropout_kwargs))
