import pdb
from sklearn.datasets import load_files
from keras.preprocessing import image
from tqdm import tqdm
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.utils import np_utils
import os
import matplotlib.pyplot as plt

def load_dataset(path):
    """
    Returns files and hot encoded target vector
    assumes that data is structured as such on disk:
    data/*/label/*.jpg
    """
    data = load_files(path)
    files = np.array(data['filenames'])
    labels = [x.split('/')[2] for x in files] # assuming data is in data/*/label/*.jpg
    labels_mapping  = {x: k for k, x in enumerate(np.unique(labels))}
    labels_categorical = [labels_mapping[x] for x in labels]
    targets = np_utils.to_categorical(np.array(labels_categorical))
    return files, targets

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def load_npy_data():
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

    return x_train, y_train, x_test, y_test

def create_model(train_tensors):

    model = Sequential()

    ### TODO: Define your architecture.
    model.add(Conv2D(input_shape=train_tensors.shape[1:],
                     filters=16,
                     kernel_size=(2, 2),
                     strides=(1, 1),
                     padding='valid',
                     activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                           strides=None,
                           padding='valid'))
    model.add(Dropout(rate=0.2))

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

    # model.add(Flatten())

    model.add(Dense(3, activation='softmax'))

    model.summary()

    return model

def compile_model(model):
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

def fit_and_save_model(model, epochs, train_tensors, train_targets,
                       valid_tensors, valid_targets):
    os.makedirs('saved_models/', exist_ok=True)
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5',
                                   verbose=1,
                                   save_best_only=True)

    model.fit(train_tensors, train_targets,
              validation_data=(valid_tensors, valid_targets),
              epochs=epochs,
              batch_size=20,
              callbacks=[checkpointer],
              verbose=1)

def plot_accuracy_history(model):
    history = model.history
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def plot_loss_history(model):
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

if __name__ == '__main__':

    files, targets = load_dataset('data/train/')
    files_val, targets_val = load_dataset('data/valid/')

    tensors = paths_to_tensor(files).astype('float32')/255
    tensors_val = paths_to_tensor(files_val).astype('float32')/255

    model = create_model(tensors)
    compile_model(model)

    fit_and_save_model(model,
                       epochs=5,
                       train_tensors=tensors,
                       train_targets=targets,
                       valid_tensors=tensors_val,
                       valid_targets=targets_val)


    plot_accuracy_history(model)
    plot_loss_history(model)

    # to do:
    # save arrays in h5 files
    # add accuracy on test data