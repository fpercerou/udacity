from __future__ import print_function
import collections
import math
import numpy as np
import random
import tensorflow as tf
from matplotlib import pylab
from six.moves import range
from sklearn.manifold import TSNE
from datetime import datetime


class Word2Vec(object):
    """l
    Class to implement Word2Vec algorithm with TensorFlow.
    vocabulary_size: size of vocabulary (less frequent are being replaced by a default token)
    batch_size: size of batch
    embedding_size: dimension of the embedding vector.
    skip_window: how many words to consider left and right.
    num_skips: how many times to reuse an input to generate a label.
    valid_size: random set of words to evaluate similarity on.
    valid_window: only pick dev samples in the head of the distribution.
    num_sampled: number of negative examples to sample.
    """
    def __init__(self, vocabulary_size, batch_size, embedding_size,
                 skip_window, num_skips, valid_size, valid_window, num_sampled,
                 log_folder):
        super(Word2Vec, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.skip_window = skip_window
        self.num_skips = num_skips
        self.valid_size = valid_size
        self.valid_window = valid_window
        self.num_sampled = num_sampled
        self.log_folder = log_folder

        # attributes initiated as None
        self.graph = None
        self.train_dataset = None
        self.train_labels = None
        self.valid_dataset = None
        self.optimizer = None
        self.loss = None
        self.normalized_embeddings = None
        self.similarity = None
        self.valid_examples = None
        self.final_embeddings = None

    @staticmethod
    def text_file_to_list(path):
        with open(path, 'rb') as f:
          text = f.read()
        return  tf.compat.as_str(text).split()

    def build_dataset(self, words, default_token='UNK'):
      """
      Build the dictionary and replace rare words with UNK token.
      words: list of words in the text
      default_token: string to replace frequent token by. (default = 'UNK')
      """
      count = [[default_token, -1]]
      count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
      dictionary = dict()
      for word, _ in count:
        dictionary[word] = len(dictionary)
      data = list()
      unk_count = 0
      for word in words:
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count = unk_count + 1
        data.append(index)
      count[0][1] = unk_count
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
      return data, count, dictionary, reverse_dictionary

    @staticmethod
    def generate_batch(data, batch_size, num_skips, skip_window, data_index):
      """
      Static method to generate a training batch for the skip-gram model.
      """
      assert batch_size % num_skips == 0
      assert num_skips <= 2 * skip_window
      batch = np.ndarray(shape=(batch_size), dtype=np.int32)
      labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
      span = 2 * skip_window + 1 # [ skip_window target skip_window ]
      buffer = collections.deque(maxlen=span)
      for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
          while target in targets_to_avoid:
            target = random.randint(0, span - 1)
          targets_to_avoid.append(target)
          batch[i * num_skips + j] = buffer[skip_window]
          labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
      return batch, labels, data_index


    def initialize(self):
        """
        Initializes the Word2Vec algorithm, populating the following attributes:
        graph, train_dataset, train_labels, valid_dataset, optimizer,
        loss, normalized_embeddings, similarity, valid_examples.
        """

        # We pick a random validation set to sample nearest neighbors. here we limit the
        # validation samples to the words that have a low numeric ID, which by
        # construction are also the most frequent.
        self.valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))

        self.graph = tf.Graph()

        with self.graph.as_default(), tf.device('/cpu:0'):

            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Variables.
            # Building the embedded representation of the words in the vocabulary.
            # From a vector of size vocabulary_size, a word is now represented as a vector of size embedding_size

            # initialization of the representation of the words (unsupervised learning)
            embeddings = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size],
                                                       -1.0,
                                                       1.0))
            # Model.
            # Look up embeddings for inputs.
            # estimating the embedded representation (smart cluster of words using the Word2Vec model)
            embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)


            # Computing the probability of a word to be in the context
            # 1 layer neural net (=logit) to compute the probability.
            softmax_weights = tf.Variable(tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                                            stddev=1.0 / math.sqrt(self.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Compute the softmax loss, using a sample of the negative labels each time.
            #
            self.loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                                           biases=softmax_biases,
                                                           inputs=embed,
                                                           labels=self.train_labels,
                                                           num_sampled=self.num_sampled,
                                                           num_classes=self.vocabulary_size))

            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
            self.normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(self.normalized_embeddings,
                                                    self.valid_dataset)
            self.similarity = tf.matmul(valid_embeddings,
                                   tf.transpose(self.normalized_embeddings))


    def run(self, data, num_steps, reverse_dictionary):
        """
        Runs the Word2Vec algorithm, populating the following attributes:
        final_embeddings.

        data: list of words in the corpus to train the Word2Vec on.
        num_steps: number of steps to iterate for the optimization
        """
        time_stamp = datetime.today().strftime("%Y%m%d_%H%M%S")

        with tf.Session(graph=self.graph) as session:
          merged_summary = tf.summary.merge_all()
          log_path = os.path.join(self.log_folder, "log_%s" % time_stamp)
          writer = tf.summary.FileWriter(log_path)
          writer.add_graph(session.graph)

          tf.global_variables_initializer().run()
          print('Initialized')
          average_loss = 0
          data_index = 0
          for step in range(num_steps):
            batch_data, batch_labels, data_index = Word2Vec.generate_batch(data=data,
                                                                           batch_size=self.batch_size,
                                                                           num_skips=self.num_skips,
                                                                           skip_window=self.skip_window,
                                                                           data_index=data_index)
            feed_dict = {self.train_dataset : batch_data,
                         self.train_labels : batch_labels}
            _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
              if step > 0:
                average_loss = average_loss / 2000
              # The average loss is an estimate of the loss over the last 2000 batches.
              print('Average loss at step %d / %d: %f' % (step, num_steps, average_loss))
              average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            if step % 10000 == 0:
              sim = self.similarity.eval()
              for i in range(self.  valid_size):
                valid_word = reverse_dictionary[self.valid_examples[i]]
                top_k = 8 # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k+1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                  close_word = reverse_dictionary[nearest[k]]
                  log = '%s %s,' % (log, close_word)
                print(log)
              print("\n\n")
          self.final_embeddings = self.normalized_embeddings.eval()
          writer.close()

    @staticmethod
    def get_tsne_representation(final_embeddings, perplexity=30, n_components=2, init='pca', n_iter=5000,
                                method='exact', num_points=400):
        """
        Returns TSNE representation of embedded words.
        """
        tsne = TSNE(perplexity=perplexity,
                    n_components=n_components,
                    init=init,
                    n_iter=n_iter,
                    method=method)
        tsne_rep = tsne.fit_transform(final_embeddings[1:num_points+1, :])
        return tsne_rep

    @staticmethod
    def plot_2d_tsne(tsne_rep, labels) :
        """
        Plots 2d tsne representation of embedded words
        """
        assert tsne_rep.shape[1] == 2, 'tsne_rep must be 2d'
        assert tsne_rep.shape[0] >= len(labels), 'More labels than embeddings'
        pylab.figure(figsize=(15,15))  # in inches
        for i, label in enumerate(labels):
            x, y = tsne_rep[i,:]
            pylab.scatter(x, y)
            pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                           ha='right', va='bottom')
        pylab.show()