import tensorflow as tf
from one_layer_covnet import OneLayerCovnet
from one_layer_fully_connected import OneLayerFullyConnected
from dropout import Dropout
from pooling import Pooling
from datetime import datetime
import numpy as np
import os
import sys
from deep_learning.utils.logger import get_standard_logger


class Covnet(object):
    """
    class to implement a convolutional neural network
    """

    _3d_one_layer = [OneLayerCovnet, Pooling, Dropout]
    _supported_one_layer = _3d_one_layer + [OneLayerFullyConnected]

    def __init__(self, batch_size, one_layers,
                 tensorboard_dir=None, log_dir='./log/'):
        Covnet.check_connectivity(one_layers)
        super(Covnet, self).__init__()
        self.batch_size = batch_size
        self.one_layers = one_layers # list of one layers network
        self.graph = tf.Graph()

        self.logger = get_standard_logger(name=Covnet.__name__,
                                          log_dir=log_dir,
                                          log_prefix='covnet',
                                          level='INFO')

        time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")
        default_tensorboard_dir = "./tensorboard/%s_%s" % (Covnet.__name__, time_stamp)
        self.tensorboard_dir = os.path.abspath(tensorboard_dir or default_tensorboard_dir)
        os.makedirs(self.tensorboard_dir)
        self.logger.debug("tensorboard dir: %s" % self.tensorboard_dir)

        with self.graph.as_default():
            self.tf_train_dataset = tf.placeholder(tf.float32,
                                                   shape=(self.batch_size,
                                                          self.first_layer.height,
                                                          self.first_layer.width,
                                                          self.first_layer.depth),
                                                   name="x")
            self.tf_train_labels = tf.placeholder(tf.float32,
                                                  shape=(self.batch_size,
                                                         self.last_layer.nb_labels),
                                                  name="y")


    @staticmethod
    def check_connectivity(one_layers):
        """
        checks that dimension between layers are correct
        """
        if not all([type(x) in Covnet._supported_one_layer for x in one_layers]):
            not_supported = [type(x) for x in one_layers if x not in Covnet._supported_one_layer ]
            raise ValueError("not supported type of one layers: %s" % not_supported)

        previous_layer = one_layers[0]
        error_log = ""
        error = False
        for i, layer in enumerate(one_layers[1:]):
            error_log_layer = "Layer %s to %s:\n" % (i, i + 1)
            error_layer = False

            if ((type(previous_layer) in Covnet._3d_one_layer)
                            & (type(layer) in Covnet._3d_one_layer)):
                condition_height = (layer.height==previous_layer.output_height)
                condition_width = (layer.width==previous_layer.output_width)
                depth_attr = getattr(previous_layer, 'conv_depth', None)  or getattr(previous_layer, 'output_depth')
                condition_depth = (layer.depth==depth_attr)

                if not condition_height:
                    error_log_layer += "height mismatch: %s != %s\n" % (layer.height, previous_layer.output_height)
                    error_layer = True
                if not condition_width:
                    error_log_layer += "width mismatch: %s != %s\n" % (layer.width, previous_layer.output_width)
                    error_layer = True
                if not condition_depth:
                    error_log_layer += "depth mismatch: %s != %s\n" % (layer.depth, depth_attr)
                    error_layer = True

            elif (type(previous_layer) in Covnet._3d_one_layer) & (type(layer)==OneLayerFullyConnected):
                depth_attr = getattr(previous_layer, 'conv_depth', None)  or getattr(previous_layer, 'output_depth')
                expected_input_dim = previous_layer.output_height * previous_layer.output_width * depth_attr
                condition_dim = (layer.input_dim==expected_input_dim)
                if not condition_dim:
                    error_log_layer += "input_dim mismatch: %s != %s\n" % (layer.input_dim, expected_input_dim)
                    error_layer = True

            elif (type(previous_layer)==OneLayerFullyConnected) & (type(layer)==OneLayerFullyConnected):
                condition_dim = (layer.input_dim==previous_layer.nb_hidden)
                if not condition_dim:
                    error_log_layer += "input_dim mismatch: %s != %s\n" % (layer.input_dim, previous_layer.nb_hidden)
                    error_layer = True

            else:
                raise NotImplemented("Connectivity not implemented between %s and %s." % (type(previous_layer), type(layer)))

            if error_layer:
                error = True
                error_log += error_log_layer + "\n\n"

            previous_layer = layer
        if error:
            raise ValueError(error_log)

    @property
    def first_layer(self):
        return self.one_layers[0]

    @property
    def last_layer(self):
        return self.one_layers[-1:][0]


    def initiate_valid_test_datasets(self, valid_dataset, test_dataset):
        with self.graph.as_default():
            tf_valid_dataset = tf.constant(valid_dataset)
            tf_test_dataset = tf.constant(test_dataset)
            return tf_valid_dataset, tf_test_dataset

    def initiate_layers_params(self):
        with self.graph.as_default():
            layers_params = [] # list of tuples (weights, biases)
            # Variables.
            for layer in self.one_layers:
                with tf.name_scope(type(layer).__name__):
                    layers_params.append(layer.initiate_weights_and_layer())
            return layers_params

    def model(self, data, layers_params):
        with self.graph.as_default():
            hidden = data
            for i, layers_param in enumerate(layers_params):
                layer_weights, layer_biases = layers_param
                layer = self.one_layers[i]

                if type(layer)==OneLayerCovnet:
                    hidden = layer.build_conv2d(hidden, layer_weights, layer_biases)

                elif  type(layer)==Pooling:
                    hidden = layer.build_pooling(hidden)

                elif  type(layer)==Dropout:
                    hidden = layer.build_dropout(hidden)

                elif type(layer)==OneLayerFullyConnected:
                    if type(prev_layer) in Covnet._3d_one_layer:
                        shape = hidden.get_shape().as_list()
                        hidden = tf.reshape(hidden, [shape[0], shape[1] * shape[2] * shape[3]])
                    hidden = layer.build_layer(hidden, layer_weights, layer_biases)

                if layer_weights is not None:
                    tf.summary.histogram("weights_%s_%s" % (type(layer).__name__, i), layer_weights)
                if layer_biases is not None:
                    tf.summary.histogram("biases_%s_%s" % (type(layer).__name__, i), layer_biases)
                tf.summary.histogram("activation_%s_%s" % (type(layer).__name__, i), hidden)

                prev_layer = layer

            return hidden


    def populate_graph(self, valid_dataset, test_dataset):
        """
        Populates graph
        """
        with self.graph.as_default():

            tf_valid_dataset, tf_test_dataset = self.initiate_valid_test_datasets(valid_dataset, test_dataset)
            layers_params = self.initiate_layers_params()


            # Training computation.
            logits = self.model(self.tf_train_dataset, layers_params)

            with tf.name_scope("cross_ent"):
                cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.tf_train_labels,
                                                                        logits=logits)
                self.loss = tf.reduce_mean(cross_entropy)
                tf.summary.scalar("loss", self.loss)

                correct_prediction = tf.equal(tf.argmax(logits, 1),
                                              tf.argmax(self.tf_train_labels, 1))
                acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                tf.summary.scalar("accuracy", acc)


            # Optimizer.
            with tf.name_scope("training"):
                self.optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(self.loss)

            # Predictions for the training, validation, and test data.
            with tf.name_scope("prediction_training"):
                self.train_prediction = tf.nn.softmax(logits)
            with tf.name_scope("prediction_validation"):
                self.valid_prediction = tf.nn.softmax(self.model(tf_valid_dataset, layers_params))
            with tf.name_scope("prediction_test"):
                self.test_prediction = tf.nn.softmax(self.model(tf_test_dataset, layers_params))


    def run_session(self, train_dataset, train_labels, valid_labels, test_labels, num_steps, num_summary_points):

        with tf.Session(graph=self.graph) as session:
          # `sess.graph` provides access to the graph used in a <a href="./api_docs/python/tf/Session"><code>tf.Session</code></a>.
          merged_summary = tf.summary.merge_all()
          writer = tf.summary.FileWriter(self.tensorboard_dir)
          writer.add_graph(session.graph)

          tf.global_variables_initializer().run()
          print('Initialized\n\n')
          for step in range(num_steps):
            offset = (step * self.batch_size) % (train_labels.shape[0] - self.batch_size)
            batch_data = train_dataset[offset:(offset + self. batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + self.batch_size), :]
            feed_dict = {self.tf_train_dataset : batch_data,
                         self.tf_train_labels : batch_labels}
            if (step % int(num_steps/np.min([num_steps, num_summary_points])) == 0):
                s = session.run(merged_summary,
                                feed_dict=feed_dict)
                writer.add_summary(s, step)
            _, l, predictions = session.run([self.optimizer, self.loss, self.train_prediction],
                                            feed_dict=feed_dict)

            if (step % int(num_steps/np.min([num_steps, 10])) == 0):
              print("step %s / %s" %(step, num_steps))
              print("Minibatch loss: %f" % l)
              print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
              print("Validation accuracy: %.1f%%" % accuracy(self.valid_prediction.eval(),
                                                             valid_labels))
              print("\n\n")
          print("Test accuracy: %.1f%%" % accuracy(self.    test_prediction.eval(), test_labels))
          writer.close()

def accuracy(predictions, labels):
    """
    simple static method to compute accuracy, not specific to Covnets
    """
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
              / predictions.shape[0])