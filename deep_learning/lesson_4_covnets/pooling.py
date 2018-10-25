import tensorflow as tf
import math

class Pooling(object):
    """
    Class to implement a 1-layer Pooling.
    """
    def __init__(self, height, width, depth, nb_labels,
                 pool_height, pool_width, pool_depth,
                 stride_height=1, stride_width=1, stride_depth=1, padding='SAME'):
        super(Pooling, self).__init__()
        self.height = height
        self.width = width
        self.depth = depth # 1: grayscale, 3: RGB
        self.nb_labels = nb_labels
        self.pool_height = pool_height
        self.pool_width = pool_width
        self.pool_depth = pool_depth
        self.stride_height = stride_height
        self.stride_width = stride_width
        self.stride_depth = stride_depth
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
        elif self.padding=='VALID':
            return math.ceil(float(self.height - self.pool_height + 1) / float(self.stride_height))
        else:
            NotImplemented("padding height not implemented for '%s' padding" % self.padding)


    @property
    def output_width(self):
        if self.padding=='SAME':
            return math.ceil(float(self.width) / float(self.stride_width))
        elif self.padding=='VALID':
            return math.ceil(float(self.width - self.pool_width + 1) / float(self.stride_width))
        else:
            NotImplemented("padding width not implemented for '%s' padding" % self.padding)

    @property
    def output_depth(self):
        if self.padding=='SAME':
            return math.ceil(float(self.depth - self.pool_depth + 1) / float(self.stride_depth))
        elif self.padding=='VALID':
            return math.ceil(float(self.depth) / float(self.stride_depth))
        else:
            NotImplemented("padding depth not implemented for '%s' padding" % self.padding)

    def initiate_weights_and_layer(self):
        """
        initiates the weight and biases tf objects
        """
        return None, None


    def build_pooling(self, data):
        """
        builds the final pooling tf object
        """
        self.check_format(data)
        return tf.nn.max_pool(data,
                              ksize=[1, self.pool_height, self.pool_width, self.pool_depth],
                              strides=[1, self.stride_height, self.stride_width, self.stride_depth],
                              padding=self.padding)
