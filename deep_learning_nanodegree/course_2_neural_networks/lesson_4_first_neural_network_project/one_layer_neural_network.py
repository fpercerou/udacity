import numpy as np
import sys
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print("Cannot import matplotlib.pyplot: %s" % e)
import pandas as pd
from functools import reduce
from utils.timeit import timeit

class OneLayerNeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate,
                 activation_function_hidden_name='sigmoid', activation_function_output_name='identity'):
        super(OneLayerNeuralNetwork, self).__init__()
        # Set number of nodes in input, hidden and output layers.
        self._input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5,
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5,
                                       (self.hidden_nodes, self.output_nodes))
        self.learning_rate = learning_rate

        self.activation_function_hidden_name = activation_function_hidden_name
        self.activation_function_output_name = activation_function_output_name
        self.activation_function_hidden = OneLayerNeuralNetwork.get_activation_function(activation_function_hidden_name)
        self.activation_function_output = OneLayerNeuralNetwork.get_activation_function(activation_function_output_name)

        self.derivative_activation_function_hidden = OneLayerNeuralNetwork.get_derivative_activation_function(activation_function_hidden_name)
        self.derivative_activation_function_output = OneLayerNeuralNetwork.get_derivative_activation_function(activation_function_output_name)

    @property
    def input_nodes(self):
        return self._input_nodes


    @staticmethod
    def get_activation_function(function_type):
        """
        Returns activation function
        """
        supported_activation_function_hidden = ['sigmoid', 'identity']
        if function_type == 'sigmoid':
            return lambda x : 1 / (1 + np.exp(-x))
        elif function_type == 'identity':
            return lambda x: x
        else:
            raise NotImplementedError("%s not implemented, only %s are"
                                      % (function_type, supported_activation_function_hidden))

    @staticmethod
    def get_derivative_activation_function(function_type):
        """
        Returns derivative of activation function.
        Takes f(x) as input!!
        for sigmoid, f'(x) = (f(x)*(1-f(x)))
        for identity f'(x) = 1
        """
        supported_activation_function_hidden = ['sigmoid', 'identity']
        if function_type == 'sigmoid':
            return lambda y : (np.multiply(y, 1-y))
        elif function_type == 'identity':
            return lambda x:  np.full(x.shape, 1)
        else:
            raise NotImplementedError("%s not implemented, only %s are"
                                      % (function_type, supported_activation_function_hidden))

    @timeit
    def train(self, X, y):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = self.get_input_shape(X)

        final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
        # Implement the backproagation function below
        delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)

    @timeit
    def get_input_shape(self, X):
        return X.shape[0]

    @timeit
    def reshape_input(self, X):
        """
        returns a matrix with self.input_nodes columns
        """
        X = np.matrix(X)
        if X.shape[1] != self.input_nodes:
            return X.T
        return X

    @timeit
    def forward_pass_train(self, X):
        ''' Implement forward pass here

            Arguments
            ---------
            X: features batch

        '''
        #### Implement the forward pass here ####
        ### Forward pass ###
        # Hidden layer - Replace these values with your calculations.
        X = self.reshape_input(X)
        hidden_inputs = self.update_hidden_input(X, np.zeros((self.get_input_shape(X), self.hidden_nodes))) # signals into hidden layer
        hidden_outputs = self.activation_function_hidden(hidden_inputs) # signals from hidden layer

        # Output layer - Replace these values with your calculations.
        final_inputs = np.dot(self.reshape_hidden_output(hidden_outputs),
                              self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_function_output(final_inputs) # signals from final output layer

        return self.reshape_final_ouput(final_outputs), self.reshape_hidden_output(hidden_outputs)

    @timeit
    def update_hidden_input(self, X, hidden_inputs):
        return np.dot(X, self.weights_input_to_hidden)

    @timeit
    def reshape_hidden_output(self, hidden_outputs):
        """
        returns a matrix with self.hidden_nodes columns
        """
        hidden_outputs = np.matrix(hidden_outputs)
        if hidden_outputs.shape[1] != self.hidden_nodes:
            return hidden_outputs.T
        return hidden_outputs

    @timeit
    def reshape_final_ouput(self, final_outputs):
        """
        returns a matrix with self.output_nodes columns
        """
        final_outputs = np.matrix(final_outputs)
        if final_outputs.shape[1] != self.output_nodes:
            return final_outputs.T
        return final_outputs

    @timeit
    def reshape_weights_input_to_hidden(self, weights_input_to_hidden):
        """
        returns a matrix with self.output_nodes columns
        """
        weights_input_to_hidden = np.matrix(weights_input_to_hidden)
        if weights_input_to_hidden.shape[1] != self.hidden_nodes:
            weights_input_to_hidden = weights_input_to_hidden.T

        if weights_input_to_hidden.shape != (self.input_nodes, self.hidden_nodes):
            raise ValueError("weights_input_to_hidden shape should be %s not %s"
                             % ((self.input_nodes, self.hidden_nodes), weights_input_to_hidden.shape))

        return weights_input_to_hidden

    @timeit
    def reshape_weights_hidden_to_output(self, weights_hidden_to_output):
        """
        returns a matrix with self.output_nodes columns
        """
        weights_hidden_to_output = np.matrix(weights_hidden_to_output)
        if weights_hidden_to_output.shape[1] != self.output_nodes:
            weights_hidden_to_output = weights_hidden_to_output.T

        if weights_hidden_to_output.shape != (self.hidden_nodes, self.output_nodes):
            raise ValueError("weights_hidden_to_output shape should be %s not %s"
                             % ((self.hidden_nodes, self.output_nodes), weights_hidden_to_output.shape))

        return weights_hidden_to_output

    @timeit
    def backpropagation(self, final_outputs, hidden_outputs, X, y):
        ''' Implement backpropagation

            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            Returns:
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''
        #### Implement the backward pass here ####
        ### Backward pass ###
        hidden_outputs = self.reshape_hidden_output(hidden_outputs)
        # Output error - Replace this value with your calculations.
        error = self.reshape_final_ouput(y) - self.reshape_final_ouput(final_outputs) # Output layer error is the difference between desired target and actual output.

        # Backpropagated error terms - Replace these values with your calculations.
        # derivative of f(x)=x activation function of last layer (if identity)
        output_error_term = np.multiply(error, self.derivative_activation_function_output(final_outputs))

        # Calculate the hidden layer's contribution to the error
        hidden_error = self.reshape_hidden_output(self.weights_hidden_to_output * output_error_term.T)

        hidden_error_term = self.reshape_hidden_output(np.multiply(hidden_error,
                                                                   self.derivative_activation_function_hidden(hidden_outputs)))

        # Weight step (input to hidden)
        delta_weights_i_h = self.reshape_weights_input_to_hidden(self.update_i_h_weight(X, hidden_error_term))

        # Weight step (hidden to output)
        delta_weights_h_o = self.reshape_weights_hidden_to_output(self.learning_rate * hidden_outputs.T * output_error_term)
        return delta_weights_i_h, delta_weights_h_o

    @timeit
    def update_i_h_weight(self, X, h_e_t):
        return self.learning_rate * self.reshape_input(X).T * h_e_t

    @timeit
    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Update weights on gradient descent step

            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        self.weights_hidden_to_output += delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden += delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step

    @timeit
    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        features = self.reshape_input(features)
        # Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = self.update_hidden_input(features, np.zeros((self.get_input_shape(features), self.hidden_nodes))) # signals into hidden layer
        hidden_outputs = self.activation_function_hidden(hidden_inputs) # signals from hidden layer

        # Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        final_outputs = self.activation_function_output(final_inputs) # signals from final output layer

        return self.reshape_final_ouput(final_outputs)

    def get_batch(self, train_features, train_targets, size):
        batch = np.random.choice(train_features.index, size=size)
        return train_features.ix[batch].values, self.get_target(train_targets.ix[batch])

    def get_target(self, targets):
        return targets['cnt']

    @staticmethod
    def get_loss_per_epoch(network, iterations, train_features,
                           train_targets, val_features, val_targets):
        """
        returns the loss function per iteration on both the train set and the validation set
        """
        losses = {'train':[], 'validation':[]}
        for ii in range(iterations+1):
            # Go through a random batch of 128 records from the training data set
            X, y = network.get_batch(train_features, train_targets, size=128)

            network.train(X, y)

            # Printing out the training progress
            train_loss = MSE(network.reshape_final_ouput(network.run(train_features)),
                             network.reshape_final_ouput(network.get_target(train_targets)))
            val_loss = MSE(network.reshape_final_ouput(network.run(val_features)),
                           network.reshape_final_ouput(network.get_target(val_targets)))
            sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii/float(iterations)) \
                             + "% ... Training loss: " + str(train_loss)[:5] \
                             + " ... Validation loss: " + str(val_loss)[:5])
            sys.stdout.flush()

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)
        losses_df = pd.DataFrame(losses)
        losses_df.index.name = 'iteration'
        return losses_df

    @staticmethod
    def plot_loss(losses, labels=None):
        """
        plots loss for the train set and the validation set
        """
        try:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            labels = labels or ['train', 'validation']
            columns = [x for x in losses.columns if any([label in x for label in labels])]
            for col in columns:
                ax.plot(losses.index, losses[col], label=col)
            ax.legend()
            # ax.ylim()
            ax.set_xlabel(losses.index.name)
            ax.set_ylabel('loss')
            plt.show()
        except NameError as e:
            print("plot_loss failed: %s" % e)

    @staticmethod
    def estimate_suitable_epoch(losses, pct_threshold, epoch_min=100):
        """
        Estimates that suitable epoch as the first iteration for which
        the validation loss has not reduced by more than pct_threshold over the window.
        Returns the max iteration if the criteria is never satisfied.
        """
        loss_copy = losses.copy()
        loss_min = loss_copy['validation'].min()
        loss_copy['improvement'] = ((loss_copy['validation'] - loss_min)
                                     / loss_min)
        loss_copy['criteria'] = ((loss_copy['improvement'] < pct_threshold)
                                  & (loss_copy.index > epoch_min))
        if any(loss_copy['criteria']):
            epoch = loss_copy[loss_copy['criteria']].index[0]
        else:
            epoch = losses.index[-1]
        print('\nsuitable epoch:', epoch)
        return epoch, losses

    @staticmethod
    def loop_learning_rate(learning_rate_range, epoch, hidden_nodes, output_nodes,
                           train_features, train_targets,
                           val_features, val_targets):
        """
        Returns loss over train and validation samples for a range of
        learning rate.
        """
        N_i = train_features.shape[1]
        losses = []
        for learning_rate in learning_rate_range:
            print('learning_rate:', np.round(learning_rate, 2))
            network = OneLayerNeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
            losse_df = OneLayerNeuralNetwork.get_loss_per_epoch(network,
                                                                epoch,
                                                                train_features,
                                                                train_targets,
                                                                val_features,
                                                                val_targets)
            print('\n')
            losses.append(losse_df.rename(columns={x: '%s_%s' % (x, np.round(learning_rate, 2))
                                                   for x in losse_df.columns}))

        losses_combined = reduce(lambda left ,right: pd.merge(left,
                                                              right,
                                                              left_index=True,
                                                              right_index=True), losses)
        return losses_combined

    @staticmethod
    def format_epoch_loss(losses, suitable_epoch, label, astype=float):
        """
        Return correctly formatted loss data at suitable_epoch
        """
        losses_epoc = losses.iloc[suitable_epoch].copy().to_frame(name='loss')
        losses_epoc = losses_epoc.reset_index().rename(columns={'index': 'label'})
        losses_epoc = losses_epoc.merge(losses_epoc['label'].str.split('_', 1, expand=True).rename(columns={0: 'type',
                                                                                                            1: label}),
                                        left_index=True, right_index=True)
        losses_epoc[label] = losses_epoc[label].astype(astype)
        return losses_epoc


    @staticmethod
    def estimate_suitable_learning_rate(losses, pct_threshold, set_type='validation'):
        """
        Estimates the suitable learning_rate as the highest learning rate for
        which the validation loss at is not more than 1 + pct_threshold of the minimum loss.
        """
        # formatting the data correctly
        losses_epoc_val = losses[losses['type']==set_type]
        min_loss = losses_epoc_val['loss'].min()
        suitable_lr = losses_epoc_val[losses_epoc_val['loss'] < min_loss * (1 + pct_threshold)]['learning_rate'].max()
        print('suitable learning rate:', suitable_lr)
        return suitable_lr

    @staticmethod
    def loop_hidden_notes(hidden_nodes_range, epoch, learning_rate, output_nodes,
                          train_features, train_targets,
                          val_features, val_targets):
        """
        Returns loss over train and validation samples for a range of
        hidden notes.
        """
        N_i = train_features.shape[1]
        losses = []
        for hidden_nodes in hidden_nodes_range:
            print('hidden_nodes:', hidden_nodes)
            network = OneLayerNeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
            losse_df = OneLayerNeuralNetwork.get_loss_per_epoch(network,
                                                                epoch,
                                                                train_features,
                                                                train_targets,
                                                                val_features,
                                                                val_targets)
            print('\n')
            losses.append(losse_df.rename(columns={x: '%s_%s' % (x, hidden_nodes) for x in losse_df.columns}))

        losses_combined = reduce(lambda left ,right: pd.merge(left,
                                                              right,
                                                              left_index=True,
                                                              right_index=True), losses)
        return losses_combined

    @staticmethod
    def estimate_suitable_hidden_notes(losses, pct_threshold, set_type='validation'):
        """
        Estimates the suitable number b of hidden_notes as the lowest number
        of hidden_notes for which the validation loss is not more than 1 + pct_threshold of the minimum loss.
        """
        # formatting the data correctly
        losses_epoc_val = losses[losses['type']==set_type]
        min_loss = losses_epoc_val['loss'].min()
        suitable_lr = losses_epoc_val[losses_epoc_val['loss'] < min_loss * (1 + pct_threshold)]['hidden_notes'].min()
        print('suitable hidden notes:', suitable_lr)
        return suitable_lr

    @staticmethod
    def set_hyperparameters(initial_epoc, initial_learning_rate, output_nodes,
                            train_features, train_targets, val_features, val_targets,
                            learning_rate_range, hidden_nodes_range, number_iter):
        """
        Static method to identify optimal hyperparameters.
        Returns the optimal hidden_nodes, epoch, learning_rate.
        It iterates over the following loop number_iter times:
        _ given the initial_epoc and the initial_learning_rate,
        sets the suitable hidden_nodes among hidden_nodes_range by running
        loop_hidden_notes and estimate_suitable_hidden_notes
        _ given the suitable hidden_nodes and the learning rate, sets the suitable
        epoch by running get_loss_per_epoch and estimate_suitable_epoch.
        _ given the suitable hidden notes and the suitable epoch, sets the suitable
        learning rate by running loop_learning_rate and estimate_suitable_learning_rate.
        """
        # initial neural network
        epoch = initial_epoc
        suitable_lr = initial_learning_rate

        for i in range(number_iter):
            print("iteration:", i)
            # first set hidden_notes
            losses_hidden = OneLayerNeuralNetwork.loop_hidden_notes(hidden_nodes_range,
                                                                    epoch,
                                                                    suitable_lr,
                                                                    output_nodes,
                                                                    train_features,
                                                                    train_targets,
                                                                    val_features,
                                                                    val_targets)
            OneLayerNeuralNetwork.plot_loss(losses_hidden, labels=['validation'])
            formatted_losses_hidden = OneLayerNeuralNetwork.format_epoch_loss(losses_hidden, epoch, 'hidden_notes', astype=int)
            hidden_nodes = OneLayerNeuralNetwork.estimate_suitable_hidden_notes(formatted_losses_hidden, 0.1)

            # then set epoch
            print("evaluating suitable epoch (epoch_max=%s)" % initial_epoc)
            N_i = train_features.shape[1]
            network = OneLayerNeuralNetwork(N_i, hidden_nodes, output_nodes, suitable_lr)
            losses = OneLayerNeuralNetwork.get_loss_per_epoch(network,
                                                              initial_epoc,
                                                              train_features,
                                                              train_targets,
                                                              val_features,
                                                              val_targets)
            OneLayerNeuralNetwork.plot_loss(losses)
            epoch, loss_details = OneLayerNeuralNetwork.estimate_suitable_epoch(losses, 0.01)

            # finally set learning rate)
            losses_lr = OneLayerNeuralNetwork.loop_learning_rate(learning_rate_range,
                                                                 epoch,
                                                                 hidden_nodes,
                                                                 output_nodes,
                                                                 train_features,
                                                                 train_targets,
                                                                 val_features,
                                                                 val_targets)
            OneLayerNeuralNetwork.plot_loss(losses_lr, labels=['validation'])
            formatted_loss_lr = OneLayerNeuralNetwork.format_epoch_loss(losses_lr, epoch, 'learning_rate')
            suitable_lr = OneLayerNeuralNetwork.estimate_suitable_learning_rate(formatted_loss_lr, 0.01)
            print("Optimal hidden_nodes:", hidden_nodes)
            print("Optimal epoch:", epoch)
            print("Optimal learning_rate:", suitable_lr)
            print("\n")

        return hidden_nodes, epoch, suitable_lr

def MSE(y, Y):
    return np.mean(np.power(y-Y, 2))

#########################################################
# Set your hyperparameters here
##########################################################
# optimal parameters found from previous static method OneLayerNeuralNetwork.set_hyperparametersOneLayerNeuralNetwork.
# hidden_nodes = 8
# learning_rate = 0.9
# iterations = 3500
# output_nodes = 1
