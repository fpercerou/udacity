from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.abstract_rnn import AbstractRNN
import tensorflow as tf

class LSTM(AbstractRNN):
    """
    Class implementing LSTM algo with tensorflow
    """
    def __init__(self, **kwargs):
        """
        _sentinelle: put there to prevent positional parameters.
                     Do not touch.
        """
        if kwargs.pop('cell_type', 'LSTM') != 'LSTM':
            raise ValueError("cell_type must be 'LSTM'")
        super(LSTM, self).__init__(cell_type='LSTM', **kwargs)

    def build_cell(self):
        # Use a basic LSTM cell
        # lstm = tf.contrib.rnn.BasicLSTMCell(self.multi_rnn_size) # deprecated
        cell = tf.nn.rnn_cell.LSTMCell(self.multi_rnn_size)

        # Add dropout to the cell
        drop = tf.contrib.rnn.DropoutWrapper(cell,
                                             output_keep_prob=self.keep_prob)
        return drop


if __name__ == '__main__':

    import numpy as np

    with open('anna.txt', 'r') as f:
        text=f.read()
    vocab = sorted(set(text))
    vocab_to_int = {c: i for i, c in enumerate(vocab)}
    int_to_vocab = dict(enumerate(vocab))
    encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

    batch_size = 10         # Sequences per batch
    n_steps = 50            # Number of sequence steps per batch
    multi_rnn_size = 128    # Size of hidden layers in LSTMs
    num_layers = 2          # Number of LSTM layers
    learning_rate = 0.01    # Learning rate
    keep_prob = 0.5         # Dropout keep probability
    epochs = 20
    grad_clip = 5
    num_layers = 2

    model = LSTM(batch_size=batch_size,
                 n_steps=n_steps,
                 multi_rnn_size=multi_rnn_size,
                 num_layers=num_layers,
                 keep_prob_value=keep_prob,
                 num_classes=len(vocab),
                 learning_rate=learning_rate,
                 grad_clip=grad_clip,
                 epochs=epochs,
                 stddev_init=0.1)

    model.run(data_x=encoded,
              data_y=encoded)
