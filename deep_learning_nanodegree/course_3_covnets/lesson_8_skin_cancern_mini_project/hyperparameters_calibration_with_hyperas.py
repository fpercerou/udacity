from __future__ import print_function

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils

from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
import keras

from keras.callbacks import TensorBoard
import tensorflow as tf
import time

raise ValueError("""
                 This attempt to wrap hyperas in a class is doomed to failed as hyperas
                 depends on two SELF CONTAINED FUNCTIONS:
                 1) one returning a train and test split (validation will be taked from train internally)
                    this one takes no argument
                 2) one running the model and only takes x_train, y_train, x_test and y_test as argument

                 This cannot be achieved by wrapping the module in a class.

                 Recommended procedure is to create a script per familly of network
                 (cnn, mlp, rnn, ...) and use an argument parser to make it flexible.

                 """)

class HyperParameterCalibrationWithHyperas(object):
    """
    Virtual Class to calibrate hyper parameters with hyperas
    """
    def __init__(self,
                 loss='categorical_crossentropy',
                 metrics=None,
                 optimizer=None,
                 log_dir='logs_template',
                 histogram_freq=0,
                 batch_size=None,
                 epochs=5,
                 verbose=2,
                 max_evals=5):
        super(HyperParameterCalibrationWithHyperas, self).__init__()
        self.loss = loss
        self.metrics = metrics or ['accuracy']
        self.optimizer = optimizer or []
        self.log_dir = log_dir
        self.histogram_freq = histogram_freq
        self.batch_size = batch_size or []
        self.epochs = epochs
        self.verbose = verbose
        self.model = None
        self.max_evals = max_evals

    def get_data(self):
        """
        Method to get train and test set splits.
        The validation will be taken internally from the train set.
        """
        raise NotImplementedError("This method must be implemented in the derived class")


    def get_model(self, x_train):
        """
        Method to get model before compilation.
        """
        raise NotImplementedError("This method must be implemented in the derived class")

    def compile_and_fit(self, x_train, y_train, x_test, y_test):
        """
        Method to compile and fit the model
        """
        self.model = self.get_model(x_train)

        self.model.compile(loss=self.loss,
                           metrics=self.metrics,
                           optimizer={{choice(self.optimizer)}})

        tensorboard = TensorBoard(log_dir="%s/model@{}".format(self.log_dir,
                                                               int(time.time())),
                                  histogram_freq=self.histogram_freq,
                                  write_graph=True,
                                  write_images=True)

        self.model.fit(x_train, y_train,
                       batch_size={{choice(self.batch_size)}},
                       epochs=self.epochs,
                       verbose=self.verbose,
                       callbacks=[tensorboard],
                       validation_data=(x_test, y_test))

        loss_and_metrics = self.model.evaluate(x_test,
                                               y_test,
                                               verbose=0)
        loss, metrics = loss_and_metrics[0], loss_and_metrics[1:]

        for i, metric in enumerate(metrics):
            print('Test %s:' % self.metrics[i], metric)

        return {**{self.metrics[i]: metric
                   for i, metric in enumerate(metrics)},
                **{'loss': -acc,
                   'status': STATUS_OK,
                   'model': self.model}}

    def optimize_hyper_parameters(self, **kwargs):
        """
        Method iterating over the set of hyper parameters.
        Return the best model and the best run
        """
        best_run, best_model = optim.minimize(model=self.compile_and_fit,
                                              data=self.get_data,
                                              algo=tpe.suggest,
                                              max_evals=self.max_evals,
                                              trials=Trials())

        x_train, y_train, x_test, y_test = self.get_data()

        print("Evalutation of best performing model:")
        evaluation = best_model.evaluate(x_test, y_test)
        print(evaluation)
        print("Best performing model chosen hyper-parameters:")
        print(pprint.pprint(best_run))
        return best_run, best_model


class HyperParameterCalibrationWithHyperasMNISTExample(HyperParameterCalibrationWithHyperas):
    """
    Virtual Class to calibrate hyper parameters on a simple model on
    MNIST data.
    """

    def get_data(self):
        """
        Data providing function:

        This function is separated from create_model() so that hyperopt
        won't reload data for each evaluation run.
        """
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        nb_classes = 10
        y_train = np_utils.to_categorical(y_train, nb_classes)
        y_test = np_utils.to_categorical(y_test, nb_classes)
        return x_train, y_train, x_test, y_test

    def get_model(self, x_train):
        """
        Model providing function:

        Create Keras model with double curly brackets dropped-in as needed.
        Return value has to be a valid python dictionary with two customary keys:
            - loss: Specify a numeric evaluation metric to be minimized
            - status: Just use STATUS_OK and see hyperopt documentation if not feasible
        The last one is optional, though recommended, namely:
            - model: specify the model just created so that we can later use it again.
        """
        model = Sequential()
        model.add(Dense(512, input_shape=(784,)))
        model.add(Activation('relu'))
        model.add(Dropout({{uniform(0, 1)}}))
        model.add(Dense({{choice([256, 512, 1024])}}))
        model.add(Activation({{choice(['relu', 'sigmoid'])}}))
        model.add(Dropout({{uniform(0, 1)}}))

        # If we choose 'four', add an additional fourth layer
        if {{choice(['three', 'four'])}} == 'four':
            model.add(Dense(100))

            # We can also choose between complete sets of layers

            model.add({{choice([Dropout(0.5), Activation('linear')])}})
            model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))

        return model


class HyperParameterCalibrationWithHyperasSKINCancer(HyperParameterCalibrationWithHyperas):
    """
    Virtual Class to calibrate hyper parameters on a model on
    skin cancer detection.
    """

    def get_data(self):
        return load_npy_data()

    def get_model(self, x_train):
        """
        Model providing function:

        Create Keras model with double curly brackets dropped-in as needed.
        Return value has to be a valid python dictionary with two customary keys:
            - loss: Specify a numeric evaluation metric to be minimized
            - status: Just use STATUS_OK and see hyperopt documentation if not feasible
        The last one is optional, though recommended, namely:
            - model: specify the model just created so that we can later use it again.

        If you get the following erro:
            File "/home/florian/anaconda3/lib/python3.6/site-packages/hyperopt/pyll/base.py", line 715, in toposort
            assert order[-1] == expr
            IndexError: list index out of range
        check that you do use double curly brackets!

        The data function has to return variables x_train, y_train, x_test, y_test.
        """
        model = Sequential()

        model.add(Conv2D(input_shape=x_train.shape[1:],
                         filters=16,
                         kernel_size=(2, 2),
                         strides=(1, 1),
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid'))
        model.add(Dropout(rate={{choice([0, 0.2])}}))

        model.add(Conv2D(filters=32,
                         kernel_size=(2, 2),
                         strides=(1, 1),
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid'))
        model.add(Dropout(rate=0.2))

        model.add(Conv2D(filters=64,
                         kernel_size=(2, 2),
                         strides=(1, 1),
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid'))
        model.add(Dropout(rate=0.2))

        model.add(Conv2D(filters=64,
                         kernel_size=(2, 2),
                         strides=(1, 1),
                         padding='valid',
                         activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2),
                               strides=None,
                               padding='valid'))
        model.add(Dropout(rate=0.2))

        model.add(GlobalAveragePooling2D())

        model.add(Dense(3, activation='softmax'))

        return model

if __name__ == '__main__':

    from skin_cancer import load_npy_data
    import pprint

    # hyper_parma_optimization = HyperParameterCalibrationWithHyperasMNISTExample(metrics=['accuracy'],
    #                                                                             optimizer=['rmsprop', 'adam', 'sgd'],
    #                                                                             batch_size=[64, 128])

    hyper_parma_optimization = HyperParameterCalibrationWithHyperasSKINCancer(metrics=['accuracy'],
                                                                              optimizer=['rmsprop'],
                                                                              batch_size=[20])


    best_run, best_model = hyper_parma_optimization.optimize_hyper_parameters()

    import pdb; pdb.set_trace()