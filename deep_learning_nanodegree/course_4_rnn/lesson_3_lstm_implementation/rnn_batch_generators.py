import numpy as np



class RNNBatchGenerator(object):
    """
    Class to produce batch generator required by any derived class
    of AbstrctRNN
    """
    def __init__(self):
        super(RNNBatchGenerator, self).__init__()

    def get_batches(self, rnn, arr_x, arr_y):
        """
        Abstract method to return the generator.
        Must be implemented in a derived class
        """
        raise NotImplementedError("must be implemented in a derived class")


class OneDSlidingGenerator(RNNBatchGenerator):
    """
    Class to produce a sliding batch generator for next item prevision.
    Returns batch of x and y where y is simply the one shifted values of x
    """
    def __init__(self):
        super(OneDSlidingGenerator, self).__init__()

    def get_batches(self, rnn, arr_x, arr_y):
        '''
        Create a generator that returns batches of size
        batch_size x n_steps from arr_x.

        Arguments
        ---------
        rnn: instance of any derived class of AstractRNN.
        arr_x: 1-d Array you want to make batches from
        arr_y: not used
        '''
        # Get the number of characters per batch and number of batches we can make
        characters_per_batch = rnn.batch_size * rnn.n_steps
        n_batches = len(arr_x) // characters_per_batch

        # Keep only enough characters to make full batches
        arr_x = arr_x[0:(n_batches * characters_per_batch)]

        # Reshape into batch_size rows
        arr_x = arr_x.reshape(rnn.batch_size, len(arr_x) // rnn.batch_size)

        for n in range(0, arr_x.shape[1], rnn.n_steps):
            # The features
            x = arr_x[:, n:n+rnn.n_steps]
            # The targets, shifted by one
            y_temp = arr_x[:, n+1:n+rnn.n_steps+1]

            # For the very last batch, y will be one character short at the end of
            # the sequences which breaks things. To get around this, I'll make an
            # array of the appropriate size first, of all zeros, then add the targets.
            # This will introduce a small artifact in the last batch, but it won't matter.
            y = np.zeros(x.shape, dtype=x.dtype)
            y[:,:y_temp.shape[1]] = y_temp
            yield x, y

class ListOfArrayGenerator(RNNBatchGenerator):
    """
    Class to produce a batch generator of arrays.
    """

    def __init__(self):
        super(ListOfArrayGenerator, self).__init__()

    def get_batches(self, rnn, arr_x, arr_y):
        n_batches = len(arr_x) // rnn.batch_size
        arr_x = arr_x[0:(n_batches * len(arr_x))]
        arr_y = arr_y[0:(n_batches * len(arr_y))]

        for n in range(0, n_batches):
            yield (np.array(arr_x[n:n+rnn.batch_size]),
                   np.array([arr_y[n:n+rnn.batch_size]]).T)




