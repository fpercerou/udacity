#coding=utf-8

from __future__ import print_function

try:
    from hyperopt import Trials, STATUS_OK, tpe
except:
    pass

try:
    from keras.datasets import mnist
except:
    pass

try:
    from keras.layers.core import Dense, Dropout, Activation
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    from keras.utils import np_utils
except:
    pass

try:
    from hyperas import optim
except:
    pass

try:
    from hyperas.distributions import choice, uniform
except:
    pass

try:
    from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
except:
    pass

try:
    from keras.layers import Dropout, Flatten, Dense
except:
    pass

try:
    from keras.models import Sequential
except:
    pass

try:
    import keras
except:
    pass

try:
    from keras.callbacks import TensorBoard
except:
    pass

try:
    import tensorflow as tf
except:
    pass

try:
    import time
except:
    pass

try:
    from skin_cancer import load_npy_data
except:
    pass

try:
    import pprint
except:
    pass
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperas.distributions import conditional

import os, numpy as np # to be callable by hyperas
npy_data_files = ['data/tensors.npy', 'data/tensors_val.npy',
                  'data/targets.npy', 'data/targets_val.npy']

if not all([os.path.exists(x) for x in npy_data_files]):

    print("All required npy data files exist, creating them...")

    files, targets = load_dataset('data/train/')
    files_val, targets_val = load_dataset('data/valid/')

    files = files[0:100]
    targets = targets[0:100]
    files_val = files_val[0:20]
    targets_val = targets_val[0:20]

    tensors = paths_to_tensor(files).astype('float32')/255
    np.save('data/tensors.npy', tensors)
    tensors_val = paths_to_tensor(files_val).astype('float32')/255
    np.save('data/tensors_val.npy', tensors_val)

    np.save('data/targets.npy', targets)
    np.save('data/targets_val.npy', targets_val)


    print("All required npy data are now exported on disk")

else:
    print("loading required npy data files from disk...")
    x_train = np.load('data/tensors.npy')
    x_test = np.load('data/tensors_val.npy')
    y_train = np.load('data/targets.npy')
    y_test = np.load('data/targets_val.npy')



def keras_fmin_fnct(space):

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
    model.add(Dropout(rate=space['rate']))

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

def get_space():
    return {
        'rate': hp.choice('rate', dropout_probs),
    }
