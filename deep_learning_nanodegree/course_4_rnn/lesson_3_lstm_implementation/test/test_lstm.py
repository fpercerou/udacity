import unittest
from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.lstm import LSTM
import tensorflow as tf
import os
import numpy as np

class TestLSTM(unittest.TestCase):

    ##########
    # Unit tests for data loading
    ##########

    def test_lstm_on_anna_data(self):

        with open(os.path.join(os.path.dirname(__file__),
                               '../anna.txt'), 'r') as f:
            text=f.read()

        text = text[:10000]

        vocab = sorted(set(text))
        vocab_to_int = {c: i for i, c in enumerate(vocab)}
        int_to_vocab = dict(enumerate(vocab))
        encoded = np.array([vocab_to_int[c] for c in text], dtype=np.int32)

        batch_size = 10         # Sequences per batch
        n_steps = 50            # Number of sequence steps per batch
        multi_rnn_size = 128         # Size of hidden layers in LSTMs
        num_layers = 2          # Number of LSTM layers
        learning_rate = 0.01    # Learning rate
        keep_prob = 0.5         # Dropout keep probability
        epochs = 2
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

        batch_loss, new_state = model.run(data_x=encoded,
                                          data_y=encoded,
                                          save_checkpoints=False)
        self.assertTrue(batch_loss < 3.15)


if __name__=='__main__':
    unittest.main()