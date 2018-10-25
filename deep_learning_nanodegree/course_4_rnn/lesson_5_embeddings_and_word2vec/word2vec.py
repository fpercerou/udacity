import time
import numpy as np
import tensorflow as tf
import utils
import random
import collections
import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pickle

print("""
      Word2Vec
      -----------
      TO DO:
      _ make folder name for checkpoint dynamic - done
      _ add argument parser
      _ add tensorboard facility (optionally to avoid writting on disk for unittest)
      _ implement the  Embedding Projector.
      _ implement early stop logic

      Documentation on embeddings (worth reading!):
      https://www.tensorflow.org/guide/embedding

      /!\\
      self.embedding_matrix is the matrix to embed inputs
      self.embedded_rep is the corresponding representaion of words in self.inputs
      in other words, self.embedded_rep is the output of the embedding layer

      """)

class Word2Vec(object):
    """
    Class meant to implement the Word2Vec algorithm

    Attributes
    ----------
    subsampling_threshold: threshold use for subsampling and removing
                           uniformative words (see method subsampling)
    window_size: size used for context around index (see method get_target)
    batch_size: batch_size used for optimization
    n_embedding: dimension of embedding vector
    stddev_init: std dev for initialization
    stddev_init: steddev of truncated normal used for variable initialization
                 defaults to 0.1
    n_sampled: num_sampled used for negative sampling of final softmax
    self.epochs: nb of iterations over the data when running the model for training
    """
    def __init__(self,
                 _sentinelle=None,
                 subsampling_threshold=None,
                 window_size=5,
                 batch_size=None,
                 n_embedding=None,
                 stddev_init=0.1,
                 n_sampled=None,
                 epochs=None,
                 checkpoint_dir=None):
        """
        _sentinelle: put there to prevent positional parameters.
                     Do not touch.
        """
        super(Word2Vec, self).__init__()
        self.subsampling_threshold = subsampling_threshold
        self.window_size = window_size
        self.batch_size = batch_size
        self.n_embedding = n_embedding
        self.stddev_init = stddev_init
        self.n_sampled = n_sampled
        self.graph = tf.Graph() # populated in build_graph
        self.epochs = epochs
        default_checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              'checkpoints')
        self.checkpoint_dir = checkpoint_dir or default_checkpoint_dir


    def preprocess_text(self, text):
        """
        Method to preprocess the corpus text (string).

        Arguments
        ---------
        text: string containing the entire corpus

        Outputs
        -------
        vocab_to_int: vocab to int dictionary
        int_to_vocab: int to vocab dictionary
        int_words: list of size of words where words have been replaced by corresponding int
        """
        words = utils.preprocess(text)
        print("Total words: {}".format(len(words)))
        print("Unique words: {}".format(len(set(words))))

        vocab_to_int, int_to_vocab = utils.create_lookup_tables(words)
        int_words = [vocab_to_int[word] for word in words]


        # saving vocab_to_int on disk
        with open(os.path.join(self.checkpoint_dir, 'vocab_to_int.pickle'), 'wb') as handle:
            pickle.dump(vocab_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return vocab_to_int, int_to_vocab, int_words

    def subsampling(self, int_words):
        """
        Method to implement subsampling for the words in int_words.
        Goes through int_words and discard each word given the probablility
        P(wi) with P(wi) = 1 - sqrt{t / f(w_i)} (f_w_i being the word frequency).
        Returs the subsampled data.

        Arguments
        ---------
        int_words: list of int (index of words)

        Outputs
        -------
        train_words: the sub-sampled list of words
        """
        nb_words = len(int_words)
        counter = collections.Counter(int_words)
        word_freq = {i: float(c) / float(nb_words)
                     for i, c in counter.items()}

        train_words = [w for w in int_words
                       if random.random() > (1 - np.sqrt(self.subsampling_threshold
                                             / word_freq[w]))]
        return train_words

    def get_target(self, words, idx):
        '''
        Method to get a list of words in a window around an index.
        Receives a list of words, an index, and given self.window size,
        then returns a list of words in the window around the index.
        For each training word, selects randomly a number R in range +/- self.window_size,
        and then use  R  words from history and  R  words from the future of the
        current word as correct labels.
        For more info: https://arxiv.org/pdf/1301.3781.pdf.


        Arguments
        ---------
        words: list of words (or list of index of words)
        idx: index of interest

        Outputs
        -------
        list of unique words in the context
        '''
        window_size_random = random.randint(1, self.window_size)
        start = idx - window_size_random if (idx - window_size_random) > 0 else 0
        end = ((idx + window_size_random + 1) if (idx + window_size_random + 1) > (len(words) - 1)
               else (len(words) - 1))

        return list(set(words[start: idx] + words[(idx + 1): start]))

    def get_batches(self, words):
        '''
        Method to get a generator of word batches as a tuple (inputs, targets).

        Arguments
        ---------
        words: list of words (or list of index of words)

        Outputs
        -------
        generator of batches of (inputs, targets) tuples
        where inputs is 1 word, and output is its corresponding context
        '''

        if len(words) < self.batch_size:
            raise ValueError("list or words too short, should be at least %s but is %s."
                             % (self.batch_size, len(words)))

        n_batches = len(words) // self.batch_size

        # only full batches
        words = words[:n_batches * self.batch_size]

        for idx in range(0, len(words), self.batch_size):
            x, y = [], []
            batch = words[idx : (idx + self.batch_size)]
            for ii in range(len(batch)):
                # for each word in the batch, adding the word and it corresponding context
                batch_x = batch[ii]
                batch_y = self.get_target(batch, ii)
                y.extend(batch_y)
                x.extend([batch_x]*len(batch_y))
            yield x, y

    def build_graph(self):
        """
        Builds self.graph, returns nothing

        """
        with self.graph.as_default():
            self.inputs = tf.placeholder(tf.int32, shape=[None], name='inputs') # shape=None to get more flexibility
            self.labels = tf.placeholder(tf.int32, shape=[None, 1], name='labels')

            n_vocab = len(self.vocab_to_int)

            ###################################################################
            # Important step
            ###################################################################
            # After this, the tensor self.embedded_rep will have shape
            # [n, embedding_size] in our example with n is self.inputs.shape[0]
            # (= total number of non unique words)
            # and contain the embeddings (dense vectors) for each of the words of self.inputs.shape.
            # As we go over the batch, the self.embedding_matrix growths (number of unique words)
            # as well as self.embedded_rep (total number of words)
            # At the end of training, self.embedding_matrix will contain the embeddings coefficient
            # for all the unique words in the vocabulary while self.embedded_rep
            # will contain the representation of each non unique word in the corpus.

            self.embedding_matrix = tf.Variable(tf.truncated_normal(shape = [n_vocab, self.n_embedding],
                                                                    mean=0.0,
                                                                    stddev=self.stddev_init),
                                                                    name='embedding_matrix')
            self.embedded_rep = tf.nn.embedding_lookup(params=self.embedding_matrix,
                                                       ids=self.inputs,
                                                       name='embedding_representation')

            softmax_w = tf.Variable(tf.truncated_normal(shape = [n_vocab, self.n_embedding],
                                                        mean=0.0,
                                                        stddev=self.stddev_init))

            softmax_b = tf.Variable(tf.truncated_normal(shape = [n_vocab],
                                                        mean=0.0,
                                                        stddev=self.stddev_init))

            # Calculate the loss using negative sampling
            loss = tf.nn.sampled_softmax_loss(weights=softmax_w,
                                              biases=softmax_b,
                                              labels=self.labels,
                                              inputs=self.embedded_rep,
                                              num_sampled=self.n_sampled,
                                              num_classes=n_vocab)

            self.cost = tf.reduce_mean(loss)
            self.optimizer = tf.train.AdamOptimizer().minimize(self.cost) # default learning rate here



    def add_validation_diagnostics_on_graph(self, valid_size=16, valid_window=100,
                                            valid_freq=1000):
        """
        Add validation diagnostics on self.graph
        Picks valid_size samples from (0, valid_size  and
        (valid_freq, valid_freq + valid_window) each ranges.
        Lower id implies more frequent.

        This is from is from Thushan Ganegedara's implementation
        It chooses a few common words and few uncommon words.
        Then, one can print out the closest words to them.
        It's a nice way to check that our embedding table is grouping together
        words with similar semantic meanings.

        Arguments
        ---------
        valid_size: Number of random set of words to evaluate similarity on
        valid_window: Size of the window to pick valid_size words from
        valid_freq: Frenquency of diagnostics

        """

        ## From Thushan Ganegedara's implementation
        with self.graph.as_default():
            self.valid_examples = np.array(random.sample(range(valid_window), valid_size//2))
            self.valid_examples = np.append(self.valid_examples,
                                            random.sample(range(valid_freq, valid_freq + valid_window),
                                                          valid_size//2))

            valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embedding_matrix), 1, keep_dims=True))
            normalized_embedding = self.embedding_matrix / norm
            valid_embedding = tf.nn.embedding_lookup(normalized_embedding, valid_dataset)
            self.similarity = tf.matmul(valid_embedding, tf.transpose(normalized_embedding))


    def run_training(self, train_words, valid_size, int_to_vocab,
                     save_checkpoints=True):
        """
        Runs training and saves checkpoint if save_checkpoints is True.

        Arguments
        ---------
        save_checkpoints: boolean to create checkpoint in the checkpoints folder
                          defaults to True
        train_words: the sub-sampled list of words
        valid_size: Number of random set of words to evaluate similarity on
        int_to_vocab: int to vocab dictionary

        Outputs:
        --------
        embeddings: the embedding matrix
        """

        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            loss = 0
            sess.run(tf.global_variables_initializer())

            nb_iteration = len(train_words) // self.batch_size
            for e in range(1, self.epochs+1):
                iteration = 1
                batches = self.get_batches(train_words)
                start = time.time()
                for x, y in batches:

                    feed = {self.inputs: x,
                            self.labels: np.array(y)[:, None]}
                    train_loss, _ = sess.run([self.cost, self.optimizer],
                                             feed_dict=feed)

                    loss += train_loss

                    if iteration % 10 == 0 or iteration==nb_iteration:
                        end = time.time()
                        print("Epoch {}/{}".format(e, self.epochs),
                              "Iteration: {}/{}".format(iteration, nb_iteration),
                              "Avg. Training loss: {:.4f}".format(loss/100),
                              "{:.4f} sec/batch".format((end-start)/100),
                              end='\r')
                        loss = 0
                        start = time.time()

                    if iteration % 1000 == 0 or iteration==nb_iteration:
                        print("")
                        ## From Thushan Ganegedara's implementation
                        # note that this is expensive (~20% slowdown if computed every 500 steps)
                        sim = self.similarity.eval()
                        for i in range(valid_size):
                            valid_word = int_to_vocab[self.valid_examples[i]]
                            top_k = 8 # number of nearest neighbors
                            nearest = (-sim[i, :]).argsort()[1:top_k+1]
                            log = 'Nearest to %s:' % valid_word
                            for k in range(top_k):
                                close_word = int_to_vocab[nearest[k]]
                                log = '%s %s,' % (log, close_word)
                            print(log)
                        print("\n\n")

                    iteration += 1
                print("")
            if save_checkpoints:
                os.makedirs(self.checkpoint_dir, exist_ok=True)
                save_path = saver.save(sess, os.path.join(self.checkpoint_dir, "word2vec.ckpt"))

            embedding = sess.run(self.embedding_matrix)
            return embedding


    def restore_network_from_disk(self):
        """
        Restores the last trained network from the checkpoint folder

        Outputs:
        --------
        embeddings: the embedding matrix
        """
        self.build_graph()
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            embedding = sess.run(self.embedding_matrix)
        return embedding


    def visualize_word_vector(self, embedding, int_to_vocab, viz_words=500,
                              interactive_mode=False):
        """
        Uses T-SNE to visualize how the high-dimensional word vectors cluster together.
        T-SNE is used to project these vectors into two dimensions while preserving local stucture.

        Arguments:
        ---------
        viz_words: number of words to visualize, defaults to 500
        int_to_vocab: int to vocab dictionary
        interactive_mode: boolean to display representation. defaults to False.
         """
        tsne = TSNE()

        embed_tsne = tsne.fit_transform(embedding[:viz_words, :])
        fig, ax = plt.subplots(figsize=(14, 14))
        for idx in range(viz_words):
            plt.scatter(*embed_tsne[idx, :], color='steelblue')
            plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)
        if interactive_mode:
            plt.show()
        file_path = os.path.join(self.checkpoint_dir, 'word_vector_rep.png')
        plt.savefig(file_path)
        print("representation saved in %s" % file_path)

    def run(self, text, valid_size=16, valid_window=100, valid_freq=1000,
            save_checkpoints=True, viz_words=500, load_from_disk=True):
        """
        Method to run the Word2Vec algorithm from the begining to the end.


        Arguments:
        ---------
        text: string containing the entire raw corpus
        Validation related arguments:
        (for more details see method add_validation_diagnostics_on_graph)
        valid_size: Number of random set of words to evaluate similarity on
                    defaults to 16
        valid_window: Size of the window to pick valid_size words from
                    defaults to 100
        valid_freq: Frenquency of diagnostics
                    defaults to 1000
        save_checkpoints: boolean to create checkpoint in the checkpoints folder
                          defaults to True
        viz_words: number of words to visualize, defaults to 500
        load_from_disk: If True loads the last checkpoint in self.checkpoint_dir.
                        If not run the whole training process.
                        Defaults to True.

        Outputs:
        --------
        embedding: embedding matrix

        """

        if not load_from_disk:
            self.vocab_to_int, int_to_vocab, int_words = self.preprocess_text(text)
            train_words = self.subsampling(int_words)

            self.build_graph()
            self.add_validation_diagnostics_on_graph(valid_size=valid_size,
                                                    valid_window=valid_window,
                                                    valid_freq=valid_freq)

            embedding = self.run_training(train_words=train_words,
                                          valid_size=valid_size,
                                          int_to_vocab=int_to_vocab,
                                          save_checkpoints=save_checkpoints)

        else:
            with open(os.path.join(self.checkpoint_dir, 'vocab_to_int.pickle'), 'rb') as handle:
                self.vocab_to_int = pickle.load(handle)
            int_to_vocab = {v: k for k, v in self.vocab_to_int.items()}
            embedding = self.restore_network_from_disk()

        if viz_words:
            self.visualize_word_vector(embedding, int_to_vocab, viz_words=500)

        return embedding

    def get_vector_rep(self, int_words):
        """
        Method to get vector representation of words.

        Arguments:
        ----------
        int_words: list of int representing words

        Outputs:
        --------
        embed: the vector representation of each word
               array of dimension (len(int_words), self.n_embedding)
               word i in int_word is represented by the row i of the output

        """
        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint(self.checkpoint_dir))
            feed = {self.inputs: int_words}
            embed = sess.run(self.embedded_rep,
                             feed_dict=feed)
        return embed

if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(__file__), 'data/text8')) as f:
        text = f.read()

    w2v = Word2Vec(subsampling_threshold=1e-5,
                   window_size=10,
                   batch_size=1000,
                   n_embedding=200,
                   stddev_init=0.1,
                   n_sampled=100,
                   epochs=20)

    embedding_matrix = w2v.run(text)

    # representation of some words:
    vocab_to_int, int_to_vocab, int_words = w2v.preprocess_text(text[:20000])
    embed_2 = w2v.get_vector_rep(int_words)
