import time
from collections import namedtuple
import numpy as np
import tensorflow as tf
from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.rnn_batch_generators import RNNBatchGenerator, OneDSlidingGenerator


print("""

      AbstractRNN
      -----------
      TO DO:
      _ INPUT TYPE (tf.int32)
      _ generalize loss function (currently softmax_cross_entropy_with_logits)
      _ generalize last layer activation function (currently softmax)
      _ make run able to load from checkpoints
      _ include evaluation on validation data in the run method
      _ generalize use of get_batches_from_1d_array to get x and y
      _ add tensorboard (making it optional in order to not have it in test model)
      _ add validation at the end of each epoch
      """)


class AbstractRNN(object):
    """
    Abstract Class implementing any type of RNN algo with tensorflow

    Attributes
    ---------
    batch_size: Batch size, the number of sequences per batch
    n_steps: Number of sequence steps per batch
    multi_rnn_size: Size of the hidden layers in the multi rnn cells block
    num_layers: Number of recursive layers in the multi rnn cells block
    batch_size: Batch size
    stddev_init: steddev of truncated normal used for variable initialization
                 defaults to 0.1
    keep_prob_value: probability used in dropout layers
    num_classes: dimension of target
    learning_rate: Learning rate for optimizer
    grad_clip: clip for gradient descent
    epochs: nb of iterations over the data when running the model for training
    rnn_batch_generator: instance of any derived class of RNNBatchGenerator
    last_output_only: use only last output. defaults to False
    input_dim: dimension of inputs. defaults to 1
    encoded_input: are inputs encoded. defaults to True
    encoded_target: are outputs encoded. defaults to False
    """
    _supported_cell_types = ['LSTM', 'GRU', 'RNN']
    _default_rnn_batch_generator = OneDSlidingGenerator()

    def __init__(self, _sentinelle=None,
                 batch_size=None,
                 n_steps=None,
                 multi_rnn_size=None,
                 num_layers=None,
                 keep_prob_value=None,
                 num_classes=None,
                 learning_rate=None,
                 grad_clip=None,
                 epochs=None,
                 stddev_init=0.1,
                 cell_type=None,
                 rnn_batch_generator=None,
                 last_output_only=False,
                 input_dim=1,
                 encoded_input=True,
                 encoded_target=False,
                 ):
        """
        _sentinelle: put there to prevent positional parameters.
                     Do not touch.
        """
        super(AbstractRNN, self).__init__()
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.multi_rnn_size = multi_rnn_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.keep_prob_value = keep_prob_value
        self.num_classes = num_classes
        self.stddev_init = stddev_init
        self.learning_rate = learning_rate
        self.grad_clip = grad_clip
        self.epochs = epochs
        assert cell_type in self._supported_cell_types, "cell_type must be in %s" % self._supported_cell_types
        self.cell_type = cell_type
        self.last_output_only = last_output_only
        self.encoded_input = encoded_input
        self.encoded_target = encoded_target

        if rnn_batch_generator is None:
            self.rnn_batch_generator = self._default_rnn_batch_generator
        else:
            self.rnn_batch_generator = rnn_batch_generator
        self.input_dim = input_dim
        self.encoded_input = encoded_input

    def build_inputs(self):
        '''
        Define placeholders for inputs, targets, and dropout
        '''
        # Declare placeholders we'll feed into the graph
        if self.input_dim == 1:
            inputs = tf.placeholder(tf.int32,
                                    shape=(self.batch_size,
                                           self.n_steps))
        else:
            inputs = tf.placeholder(tf.float32,
                                    shape=(None, # self.batch_size,
                                           self.n_steps,
                                           self.input_dim))
        if not self.last_output_only:
            # will be reshape in build_loss
            targets = tf.placeholder(tf.int32,
                                     shape=(self.batch_size, self.n_steps))
        else:
            targets = tf.placeholder(tf.int32,
                                     shape=(None, # self.batch_size,
                                            1))

        # Keep probability placeholder for drop out layers
        keep_prob = tf.placeholder(tf.float32)

        return inputs, targets, keep_prob

    def build_cell(self):
        raise NotImplementedError("build_cell to be implemented in derived classed")

    def build_multi_rnn_block(self):
        '''
        Build the multi rnn block as one cell.
        '''
        # Stack up multiple recurcive layers, for deep learning
        cell = tf.contrib.rnn.MultiRNNCell([self.build_cell()
                                            for _ in range(self.num_layers)])
        return cell


    def build_output(self, multi_rnn_block_output):
        '''
        Build a softmax layer, return the softmax output and logits.
        Arguments
        ---------
        multi_rnn_block_output: List of output tensors from the multi rnn layer block
        '''

        # Reshape output so it's a bunch of rows, one row for each step for each sequence.
        # Concatenate multi_rnn_block_output over axis 1 (the columns)
        if not self.last_output_only:
            seq_output = tf.concat(multi_rnn_block_output, 1)
            # Reshape seq_output to a 2D tensor with multi_rnn_size columns
            x = tf.reshape(seq_output, (-1, seq_output.shape[2])) # seq_output.shape[2] = multi_rnn_size

        else:
            x = multi_rnn_block_output
        # Connect the RNN outputs to a softmax layer
        with tf.variable_scope('softmax'):
            # Create the weight and bias variables here
            softmax_w = tf.Variable(tf.truncated_normal(shape=(self.multi_rnn_size,
                                                               self.num_classes),
                                                        stddev=self.stddev_init)) # /!\ hard coded value /!\
            softmax_b = tf.Variable(tf.truncated_normal(shape=(1, self.num_classes)))

        # Since output is a bunch of rows of RNN cell outputs, logits will be a bunch
        # of rows of logit outputs, one for each step and sequence
        logits = tf.add(tf.matmul(x, softmax_w), softmax_b)

        # Use softmax to get the probabilities for predicted characters
        if self.num_classes==1:
            out = tf.nn.sigmoid(logits)
        else:
            out = tf.nn.softmax(logits)

        return out, logits


    def build_loss(self, logits, targets):
        '''
        Calculate the loss from the logits and the targets.
        Arguments
        ---------
        logits: Logits from final fully connected layer
        targets: Targets for supervised learning
        '''

        # One-hot encode targets and reshape to match logits, one row per sequence per step
        if not self.encoded_target:
            y_one_hot = tf.one_hot(self.targets, self.num_classes)
            y_reshaped = tf.reshape(y_one_hot, (-1, self.num_classes))
        else:
            y_reshaped = targets

        # Softmax cross entropy loss
        if self.num_classes ==1:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_reshaped,
                                                                          logits=logits))
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_reshaped,
                                                                          logits=logits))

        return loss, y_reshaped

    def build_optimizer(self, loss):
        '''
        Build optmizer for training, using gradient clipping.
        Arguments:
        loss: Network loss
        '''

        # Optimizer for training, using gradient clipping to control exploding gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), self.grad_clip)
        train_op = tf.train.AdamOptimizer(self.learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))

        return optimizer

    def build_model(self):
        """
        Build model ready to be run
        """

        # fp commented out:
        # # When we're using this network for sampling later, we'll be passing in
        # # one character at a time, so providing an option for that
        # if sampling == True:
        #     batch_size, n_steps = 1, 1
        # else:
        #     batch_size, n_steps = self.batch_size, self.n_steps

        tf.reset_default_graph()

        # Build the input placeholder tensors
        self.inputs, self.targets, self.keep_prob = self.build_inputs()

        # Build the multi rnn block
        self.cell = self.build_multi_rnn_block()
        self.initial_state = self.cell.zero_state(self.batch_size,
                                                 tf.float32)

        ### Run the data through the RNN layers
        # First, one-hot encode the input tokens
        if self.encoded_input:
            x_one_hot = tf.one_hot(self.inputs, self.num_classes)
        else:
            x_one_hot = self.inputs

        # Run each sequence step through the RNN with tf.nn.dynamic_rnn
        outputs, state = tf.nn.dynamic_rnn(self.cell, x_one_hot,
                                           # initial_state=self.initial_state,
                                           dtype=tf.float32)
        self.final_state = state

        if self.last_output_only:
            outputs = outputs[:, -1]

        # Get softmax predictions and logits
        self.prediction, self.logits = self.build_output(outputs)

        # Loss and optimizer (with gradient clipping)
        self.loss, targets = self.build_loss(self.logits,
                                             self.targets)
        self.optimizer = self.build_optimizer(self.loss)

        if self.num_classes==1:
            self.binary_prediction = tf.cast(tf.scalar_mul(2, self.prediction),
                                             tf.int32)
        else:
            self.binary_prediction = tf.argmax(self.prediction, 1)
            targets = tf.argmax(targets, 1)

        correct_pred = tf.equal(tf.cast(self.binary_prediction, tf.int32),
                                tf.cast(targets, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def run(self, data_x, data_y, val_x, val_y, print_every_n=10, save_every_n=200,
            save_checkpoints=True, use_previous_state=True):
        """
        Runs the model and saves checkpoint on disk
        """
        assert len(data_x) == len(data_y), "data_x and data_y must have same length"

        self.build_model()

        saver = tf.train.Saver(max_to_keep=100)
        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(),
                              tf.local_variables_initializer()))

            # Use the line below to load a checkpoint and resume training
            # saver.restore(sess, 'checkpoints/______.ckpt')

            if self.last_output_only:
                nb_batch = len(data_x) // (self.batch_size)
            else:
                nb_batch = len(data_x) // (self.batch_size * self.n_steps)
            for e in range(self.epochs):
                counter = 0
                # Train network
                new_state = sess.run(self.initial_state)
                acc_ls = []
                loss_ls = []
                for x, y in self.rnn_batch_generator.get_batches(self, data_x, data_y):
                    counter += 1
                    start = time.time()

                    feed = {self.inputs: x,
                            self.targets: y,
                            self.keep_prob: self.keep_prob_value}

                    if use_previous_state:
                        feed[self.initial_state] = new_state

                    bacth_loss , new_state, _, batch_acc = sess.run([self.loss,
                                                               self.final_state,
                                                               self.optimizer,
                                                               self.accuracy],
                                                               feed_dict=feed)

                    acc_ls.append(batch_acc)
                    loss_ls.append(bacth_loss)
                    acc = np.mean(acc_ls)
                    loss = np.mean(loss_ls)
                    if (counter % print_every_n == 0):
                        end = time.time()
                        print('Epoch: {}/{}... '.format(e+1, self.epochs),
                              'Training Step: {}/{}... '.format(counter, nb_batch),
                              'Training loss: {:.4f}... '.format(loss),
                              'Training accuracy: {:.4f}... '.format(acc),
                              '{:.4f} sec/batch'.format((end-start)),
                              end='\r')

                    if counter == nb_batch:
                        val_acc = []
                        val_loss = []
                        val_state = sess.run(self.cell.zero_state(self.batch_size,
                                                                  tf.float32))
                        for x, y in self.rnn_batch_generator.get_batches(self, val_x, val_y):
                            feed = {self.inputs: x,
                                    self.targets: y,
                                    self.keep_prob: 1,
                                    self.initial_state: val_state}
                            batch_loss, batch_acc, val_state = sess.run([self.loss, self.accuracy, self.final_state], feed_dict=feed)
                            val_acc.append(batch_acc)
                            val_loss.append(batch_loss)

                        print("")
                        print('Validation loss: {:.4f}... '.format(np.mean(val_loss)),
                              'Validation accuracy: {:.4f}... '.format(np.mean(val_acc)))


                    if ((counter * (e + 1)) % save_every_n == 0):
                        if save_checkpoints:
                            saver.save(sess, "checkpoints/e{}_i{}_l{}.ckpt".format(e, counter,
                                                                                   self.multi_rnn_size))
            if save_checkpoints:
                saver.save(sess, "checkpoints/e{}_i{}_l{}.ckpt".format(e, counter, self.multi_rnn_size))
            return loss

