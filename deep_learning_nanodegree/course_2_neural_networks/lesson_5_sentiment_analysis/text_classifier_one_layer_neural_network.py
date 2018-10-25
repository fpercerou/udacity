from deep_learning_nanodegree.course_2_neural_networks.lesson_4_first_neural_network_project.one_layer_neural_network import OneLayerNeuralNetwork
from collections import Counter
import numpy as np
import numba as nb
from numba.types import List, int64
from utils.timeit import timeit

# raise ValueError("This class is not completed yet")

class TextClassifierOneLayerNeuralNetwork(OneLayerNeuralNetwork):
    def __init__(self, train_reviews, train_labels,
                 polarity_cutoff, min_count,
                 activation_function_hidden_name='sigmoid',
                 activation_function_output_name='sigmoid',
                 positive_label='POSITIVE', input_nodes=None,
                 **kwargs):
        self.word2index = {}
        self.label2index = {}
        self.review_vocab = []
        self.label_vocab = []
        self.positive_label = positive_label
        self.polarity_cutoff = polarity_cutoff
        self.min_count = min_count
        TextClassifierOneLayerNeuralNetwork.initiate_counters(self)
        self.pre_process_data(train_reviews, train_labels, self.polarity_cutoff, self.min_count)

        super().__init__(activation_function_hidden_name=activation_function_hidden_name,
                         activation_function_output_name=activation_function_output_name,
                         input_nodes=None,
                         **kwargs)

    @staticmethod
    def initiate_counters(obj):
        obj.positive_counts = Counter()
        obj.negative_counts = Counter()
        obj.total_counts = Counter()
        obj.pos_neg_ratios = Counter()

    @property
    def input_nodes(self):
        return self.review_vocab_size

    @property
    def review_vocab_size(self):
        return len(self.review_vocab)

    @property
    def label_vocab_size(self):
        return len(self.label_vocab)

    def populate_counters(self, reviews, labels):
        """
        Populate the counters.
        """
        for i in range(len(reviews)):
            if(labels[i] == self.positive_label):
                for word in reviews[i].split(" "):
                    self.positive_counts[word] += 1
                    self.total_counts[word] += 1
            else:
                for word in reviews[i].split(" "):
                    self.negative_counts[word] += 1
                    self.total_counts[word] += 1

        for term, cnt in list(self.total_counts.most_common()):
            if(cnt >= 50): # TO DO: make this 50 "parameterizable"
                pos_neg_ratio = self.positive_counts[term] / float(self.negative_counts[term]+1)
                self.pos_neg_ratios[term] = pos_neg_ratio

        for word,ratio in self.pos_neg_ratios.most_common():
            if(ratio > 1):
                self.pos_neg_ratios[word] = np.log(ratio)
            else:
                self.pos_neg_ratios[word] = -np.log((1 / (ratio + 0.01)))


    def populate_review_vocab(self, reviews):
        """
        Populate review_vocab with all of the words in the given reviews.
        """
        TextClassifierOneLayerNeuralNetwork.initiate_counters(self)
        self.populate_counters(reviews, labels)
        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                ## New for Project 6: only add words that occur at least min_count times
                #                     and for words with pos/neg ratios, only add words
                #                     that meet the polarity_cutoff
                if(self.total_counts[word] > self.min_count):
                    if(word in self.pos_neg_ratios.keys()):
                        if((self.pos_neg_ratios[word] >= self.polarity_cutoff)
                            or (self.pos_neg_ratios[word] <= -self.polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)
        # Convert the vocabulary set to a list so we can access words via indices
        self.review_vocab = list(review_vocab)

    def populate_label_vocab(self, labels):
        """
        Populate label_vocab with all of the words in the given labels.
        """
        label_vocab = set()
        for label in labels:
            label_vocab.add(label)
        # Convert the label vocabulary set to a list so we can access labels via indices
        self.label_vocab = list(label_vocab)


    def populate_word2index(self):
        """
        Returns dictionary of words in the vocabulary mapped to index positions
        """
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i

    def populate_label2index(self):
        """
        Returns  a dictionary of labels mapped to index positions
        """
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

        """
        Populates counters, review_vocab, label_vocab, word2index, label2index
        """
    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):
        self.populate_review_vocab(reviews)
        self.populate_label_vocab(labels)
        self.populate_word2index()
        self.populate_label2index()

    @timeit
    def reshape_input(self, reviews_raw):
        """
        /!\\ IMPORTANT /!\\
        X is the list of strings of reviews
        """
        if (not isinstance(reviews_raw, list)
            or not isinstance(reviews_raw[0], list)
            or not isinstance(reviews_raw[0][0], int)):
            reviews_reshaped = list()
            for review in reviews_raw:
                indices = set()
                for word in review.split(" "):
                    if(word in self.word2index.keys()):
                        indices.add(self.word2index[word])
                reviews_reshaped.append(list(indices))
            return reviews_reshaped
        else:
            return reviews_raw

    @timeit
    def update_hidden_input(self, reshape_x, hidden_input):
        return update_hidden_input_nojit(np.array(range(len(reshape_x))),
                                         reshape_x,
                                         self.weights_input_to_hidden,
                                         hidden_input)

    @timeit
    def update_i_h_weight(self, reshape_x, h_e_t):
        return update_i_h_weight_nojit(np.array(range(len(reshape_x))),
                                       reshape_x,
                                       self.learning_rate * h_e_t,
                                       self.weights_input_to_hidden)

    @timeit
    def get_batch(self, reshape_train_features, train_targets, size):
        batch = np.random.choice(range(len(reshape_train_features)), size=size)
        feature_batch = [reshape_train_features[index] for index in batch]
        target_batch = [train_targets[index] for index in batch]
        return feature_batch, target_batch

    @timeit
    def get_input_shape(self, X):
        return len(X)

    @timeit
    def get_target(self, targets):
        return targets


# To do: make jit to work
@nb.jit(nb.f8[:, :](nb.i4[:], List(List(int64, reflected=True), reflected=True), nb.f8[:, :], nb.f8[:, :]))
def update_hidden_input_jit(range_index, reshape_x, weights, hidden_input):
    for k in range_index:
        for index in reshape_x[k]:
            hidden_input[k, :] = add_vector(hidden_input[k, :], weights[index, :])
    return hidden_input

# To do: make jit to work
@nb.jit(nb.f8[:](nb.f8[:], nb.f8[:]))
def add_vector(h, w):
    for j in range(len(h)):
        h[j] += w[j]
    return h

def update_hidden_input_nojit(range_index, reshape_x, weights, hidden_input):
    for k in range_index:
        for index in reshape_x[k]:
            hidden_input[k, :] += weights[index, :]
    return hidden_input

def update_i_h_weight_nojit(range_index, reshape_x, h_e_t, weights):
    delta_weights = np.zeros(weights.shape)
    for k in range_index:
        for index in reshape_x[k]:
            delta_weights[index, :] = np.add(delta_weights[index, :], h_e_t[k, :])
    return delta_weights

def get_target_for_label(label, positive_label='POSITIVE'):
    return [int(l==positive_label) for l in label]

if __name__ =='__main__':

    g = open('reviews.txt','r') # What we know!
    reviews = list(map(lambda x:x[:-1],g.readlines()))
    g.close()

    g = open('labels.txt','r') # What we WANT to know!
    labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
    g.close()

    sent = TextClassifierOneLayerNeuralNetwork(train_reviews=reviews,
                                      train_labels=labels,
                                      hidden_nodes=2,
                                      output_nodes=1,
                                      learning_rate=1, # will be scaled by batch size
                                      polarity_cutoff=0.05,
                                      min_count=20)

    r_X = sent.reshape_input(reviews)
    iterations = 10

    losse_df = OneLayerNeuralNetwork.get_loss_per_epoch(sent,
                                                        iterations,
                                                        train_features=r_X[1000:], # r_X[:-1000],
                                                        train_targets=get_target_for_label(labels[1000:]), # get_target_for_label(labels[get_target_for_label(labels[:-1000]), # :-1000]),
                                                        val_features=r_X[:1000],
                                                        val_targets=get_target_for_label(labels[:1000]))
    import pdb; pdb.set_trace()

    # import time
    # start = time.time()
    # test = sent.update_hidden_input(r_X,
    #                                 np.zeros((len(r_X), sent.hidden_nodes)))
    # end = time.time()
    # print(end - start)

    # import time
    # start = time.time()
    # test2 = update_hidden_input_jit(np.array(range(len(r_X))),
    #                                 r_X,
    #                                 sent.weights_input_to_hidden,
    #                                 np.zeros((len(r_X ), sent.hidden_nodes)))
    # end = time.time()
    # print(end - start)
