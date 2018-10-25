from talos import Scan, Reporting
from talos.model.layers import hidden_layers
from talos.model import lr_normalizer
from talos.metrics.keras_metrics import fmeasure
from keras.layers import GlobalAveragePooling2D # , Conv2D, MaxPooling2D
from keras.layers import Dropout, Dense #, Flatten
from keras.models import Sequential

raise ValueError("This module uses talos, better use hyperas")


def model_from_params(x_train, y_train, x_val, y_val, params):
    """
    Build, compile and fits a keras model from a dictionary of params.
    Returns the model object and its corresponding history: history, model
    """

    # next first we build the model exactly like we would normally do it
    model = Sequential()
    model.add(Dense(10, input_dim=x_train.shape[1],
                    activation=params['activation'],
                    kernel_initializer='normal'))

    model.add(Dropout(params['dropout']))

    # if we want to also test for number of layers and shapes, that's possible
    hidden_layers(model, params, 1)

    # then we finish again with completely standard Keras way
    model.add(Dense(1, activation=params['last_activation'],
                    kernel_initializer='normal'))

    model.compile(loss=params['losses'],
                  # here we add a regulizer normalization function from Talos
                  optimizer=params['optimizer'](learning_rate=lr_normalizer(params['lr'],
                                                                            params['optimizer'])),
                  metrics=['acc', fmeasure])

    history = model.fit(x_train, y_train,
                        validation_data=[x_val, y_val],
                        batch_size=params['batch_size'],
                        epochs=params['epochs'],
                        verbose=0)

    # finally we have to make sure that history object and model are returned
    return history, model

if __name__ == '__main__':

    from skin_cancer import load_npy_data
    import tensorflow as tf

    tensors, tensors_val, targets, targets_val = load_npy_data()

    # parameter space:

    # if tuple must be of type (min, max, number of values)
    # (mast will not be included (np.arange called in backend))
    # see method param_format in anaconda3/lib/python3.6/site-packages/talos/parameters/handling.py
    # this method is called in the __init__ of Scan

    # You cannot mix keras object and pure tf objects !
    # https://stackoverflow.com/questions/50056356/could-not-interpret-optimizer-identifier-error-in-keras

    p = {'lr': (0, 0.2, 1), # (0.5, 5, 10),
         'first_neuron': [4], # [4, 8, 16, 32, 64],
         'hidden_layers': [2], # [0, 1, 2],
         'batch_size': (10, 15, 1), # (2, 30, 10),
         'epochs': [150],
         'dropout': (0, 0.2, 2), # (0, 0.5, 5),
         'weight_regulizer':[None],
         'emb_output_dims': [None],
         'shape': ['brick','long_funnel'],
         'optimizer': [tf.train.AdamOptimizer],
                    # [tf.keras.optimizers.Nadam, tf.keras.optimizers.RMSprop],
         'losses': [tf.losses.softmax_cross_entropy],
                    # [tf.keras.losses.logcosh]
         'activation':[tf.nn.relu], # [tf.nn.elu],
         'last_activation': [tf.nn.sigmoid]}

    scan = Scan(x=tensors,
                y=targets,
                model=model_from_params,
                # grid_downsample=None,
                params=p,
                dataset_name='skin_cancer',
                experiment_no='1')

    r = Reporting("skin_cancer_f.csv")

    import pdb; pdb.set_trace()
