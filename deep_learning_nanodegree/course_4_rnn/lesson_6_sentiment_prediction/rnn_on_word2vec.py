from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.abstract_rnn import AbstractRNN
from deep_learning_nanodegree.course_4_rnn.lesson_5_embeddings_and_word2vec.word2vec import Word2Vec
import tensorflow as tf
import tqdm
import numpy as np
import re
import pickle

print("""

      RnnOnWord2Vec
      -----------
      As of 03/10/2028,

      we have found that directly plugging an embedding matrix to the newtork with
      using a word2vec representation performs better.

      This is probably due to a weak word2vec representation.

      Should you want to implement a direct embedding, please refer to the
      jupyter notebook Sentiment_RNN.ipynb.

      """)

class RnnOnWord2Vec(object):
    """
    Class meant to plug an AbstractRNN to a Word2Vec

    Attributes
    ----------
    rnn: instance of type any derived class of AbstractRNN
    w2v: instance of type Word2Vec
    """
    def __init__(self, rnn, w2v):
        super(RnnOnWord2Vec, self).__init__()
        self.rnn = rnn
        self.w2v = w2v

    def run(self, w2v_text, rnn_list_of_text, labels,
            w2v_valid_size=16, w2v_valid_window=100, w2v_valid_freq=1000,
            w2v_save_checkpoints=True, w2v_viz_words=500, rnn_print_every_n=10,
            rnn_save_every_n=200, rnn_save_checkpoints=True, path_to_features=None):
        """
        Method meant to run first the w2v model from the corpus of the text
        """

        # estimating w2v model
        self.w2v.run(text=w2v_text,
                     valid_size=w2v_valid_size,
                     valid_window=w2v_valid_window,
                     valid_freq=w2v_valid_freq,
                     save_checkpoints=w2v_save_checkpoints,
                     viz_words=w2v_viz_words)

        features = self.get_array_rep(rnn_list_of_text,
                                      path_to_features=path_to_features)

        self.rnn.run(features[:len(features)-1000],
                     labels[:len(features)-1000],
                     features[len(features)-1000:],
                     labels[len(features)-1000:],
                     print_every_n=rnn_print_every_n,
                     save_every_n=rnn_save_every_n,
                     save_checkpoints=rnn_save_checkpoints)

        return features

    def get_array_rep(self, list_of_text, path_to_features=None):
        """
        Method to get the array representation of the list of text.self

        Arguments:
        ----------
        list_of_text: list of strings. Each string being a review / an article
                      from which we want to get a prediction
        path_to_features: path to pickle of features array.
                          if exists, features are loaded directly from there
                          if not features are computed and then stored
                          defaults to None

        Outputs:
        --------
        features: list of arrays.
                  Each array is the matrix representation of the corresponding
                  string.
                  Each array is of shape (self.rnn.n_steps, self.w2v.n_embedding)
                  (fix number of words in a review / article by embedding dimension)
                  if a review has less than self.rnn.n_steps words then the first
                  rows will be only zeros
                  if a review has more than self.rnn.n_steps, we only take the first
                  self.rnn.n_steps of the review
        """

        if path_to_features and os.path.exists(os.path.join(path_to_features)):
            print("loading features from %s" % path_to_features)
            with open(path_to_features, 'rb') as handle:
                features = pickle.load(handle)

        else:
            print("Splitting list of strings into list of list of words...")
            list_list_of_words = [re.split(' +', review) for review in tqdm.tqdm(list_of_text)]

            print("Converting list of list of words into list of list of ints...")
            rnn_int_words = [[self.w2v.vocab_to_int[word]
                              for word in list_of_words[: (np.min([len(list_of_words), self.rnn.n_steps]) - 1)]
                              if word in self.w2v.vocab_to_int.keys()]
                             for list_of_words in tqdm.tqdm(list_list_of_words)]


            features = []
            print("Loading word2vec representation of each word of each string")

            # flattening the list of list into one list
            flat_rnn_int_words = [item for sublist in rnn_int_words for item in sublist]

            # get the vector representation of all the vectors in one go (v fast)
            embbed_flat = self.w2v.get_vector_rep(flat_rnn_int_words)

            # get the index of each review to unflat the list
            counter = 0
            flat_rnn_int_index = []
            for i in rnn_int_words:
                flat_rnn_int_index.append((counter, counter + len(i) - 1))
                counter += len(i)

            # now we have list of list of word vector representations
            embbed = [embbed_flat[s:e+1] for s,e in flat_rnn_int_index]

             # adding necessary number of rows (containing zeros)
            features = []
            for e in tqdm.tqdm(embbed):
                zeros = np.zeros((self.rnn.n_steps - e.shape[0], self.w2v.n_embedding))
                features.append(np.append(zeros, e, axis=0))

            if path_to_features:
                print("saving features to %s" % path_to_features)
                with open(path_to_features, 'wb') as handle:
                    pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # list of arrays (nb_reviews * self.rnn.n_steps * self.w2v.n_embedding)
        return features

if __name__ == '__main__':

    from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.lstm import LSTM
    from deep_learning_nanodegree.course_4_rnn.lesson_3_lstm_implementation.rnn_batch_generators import ListOfArrayGenerator
    import os


    with open('sentiment-network/reviews.txt', 'r') as f:
        reviews = f.read()

    with open('sentiment-network/labels.txt', 'r') as f:
        labels = f.read()
    labels = labels.split('\n')
    labels = np.array([1 if each == 'positive' else 0 for each in labels])

    from string import punctuation
    all_text = ''.join([c for c in reviews if c not in punctuation])
    review_ls = all_text.split('\n') # list of reviews


    with open(os.path.join(os.path.dirname(__file__), '../lesson_5_embeddings_and_word2vec/data/text8')) as f:
        text = f.read()

    w2v = Word2Vec(subsampling_threshold=1e-5,
                   window_size=10,
                   batch_size=1000,
                   n_embedding=200,
                   stddev_init=0.1,
                   n_sampled=100,
                   epochs=20)

    model = LSTM(batch_size=100,         # Sequences per batch
                 n_steps=50,            # Number of sequence steps per batch
                 multi_rnn_size=32,    # Size of hidden layers in LSTMs
                 num_layers=2,          # Number of LSTM layers
                 learning_rate=0.01,    # Learning rate
                 keep_prob_value=0.2print(""""

      AbstractRNN
      -----------
      TO DO:
,   # Dropout keep probability
                 epochs=20,
                 grad_clip=5,
                 stddev_init=0.1,
                 num_classes=1,
                 last_output_only=True,
                 rnn_batch_generator=ListOfArrayGenerator(),
                 input_dim=w2v.n_embedding,
                 encoded_input=False,
                 encoded_target=False)

    rnw = RnnOnWord2Vec(w2v=w2v,
                        rnn=model)

    features = rnw.run(w2v_text=text,
                       rnn_list_of_text=review_ls,
                       labels=labels,
                       path_to_features=None) # os.path.abspath(os.path.join(os.path.dirname(__file__), 'embedded_features.pickle'))
