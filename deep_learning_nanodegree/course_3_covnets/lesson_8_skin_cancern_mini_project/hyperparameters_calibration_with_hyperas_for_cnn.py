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

def create_model_template(x_train, y_train, x_test, y_test):
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

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    tensorboard = TensorBoard(log_dir="logs_template/model@{}".format(int(time.time())),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    model.fit(x_train, y_train,
              batch_size={{choice([64, 128])}},
              epochs=5,
              verbose=2,
              callbacks=[tensorboard],
              validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

def create_model(x_train, y_train, x_test, y_test, dropout_probs=[0, 0.2]):
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
    model.add(Dropout(rate={{choice(dropout_probs)}}))

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

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())),
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    # tf.global_variables_initializer()
    model.fit(x_train, y_train,
              validation_data=(x_test, y_test),
              epochs=5,
              batch_size=20,
              callbacks=[tensorboard],
              verbose=2)
    score, acc = model.evaluate(x_test, y_test, verbose=0)

    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}


if __name__ == '__main__':

    from skin_cancer import load_npy_data
    import pprint

    best_run, best_model = optim.minimize(model=create_model,
                                          data=load_npy_data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials(),
                                          debug_mode=True)

    x_train, y_train, x_test, y_test = load_npy_data() # should be test not val

    print("Evalutation of best performing model:")
    evaluation = best_model.evaluate(x_test, y_test)
    print(evaluation)
    print("Best performing model chosen hyper-parameters:")
    print(pprint.pprint(best_run))