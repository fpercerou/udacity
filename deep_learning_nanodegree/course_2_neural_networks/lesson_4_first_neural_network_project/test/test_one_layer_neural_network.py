import unittest
import numpy as np
from deep_learning_nanodegree.course_2_neural_networks.lesson_4_first_neural_network_project.one_layer_neural_network import OneLayerNeuralNetwork
import pandas as pd
from utils.temp_seed import temp_seed
import os
# TO DO: add unittest for set_hyperparameters

inputs = np.array([[0.5, -0.2, 0.1]])
targets = np.array([[0.4]])
test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])
folder = os.path.dirname(__file__)
data_path = os.path.join(folder, '../Bike-Sharing-Dataset/hour.csv')



class TestOneLayerNeuralNetwork(unittest.TestCase):

    ##########
    # Unit tests for data loading
    ##########

    # def test_data_path(self):
    #     # Test that file path to dataset has been unaltered
    #     self.assertTrue(data_path.lower() == os.path.join(folder, '../Bike-Sharing-Dataset/hour.csv'))

    def test_data_loaded(self):
        # Test that data frame loaded
        rides = pd.read_csv(data_path)
        self.assertTrue(isinstance(rides, pd.DataFrame))

    ##########
    # Unit tests for network functionality
    ##########

    def test_activation(self):
        network = OneLayerNeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function_hidden(0.5) == 1/(1+np.exp(-0.5))))

    def test_train(self):
        # Test that weights are updated correctly on training
        network = OneLayerNeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))

    def test_run(self):
        # Test correctness of run method
        network = OneLayerNeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


    def test_get_loss_per_epoch(self):

        # importing data for unittest
        train_features = pd.read_hdf(os.path.join(folder, 'data/test_train_features.h5'), key='df')
        train_targets = pd.read_hdf(os.path.join(folder, 'data/test_train_targets.h5'), key='df')
        val_features = pd.read_hdf(os.path.join(folder, 'data/test_val_features.h5'), key='df')
        val_targets = pd.read_hdf(os.path.join(folder, 'data/test_val_targets.h5'), key='df')
        # results to reproduce
        losse_df_to_match = pd.read_hdf(os.path.join(folder, 'data/test_losse_df.h5'), key='df')

        with temp_seed(5): # to ensure that we can reproduce results
            # optimal parameters found from previous static method OneLayerNeuralNetwork.set_hyperparameters.
            output_nodes = 1  # defining again in case we didnt run previous cell
            hidden_nodes = 8
            learning_rate = 0.9
            iterations = 100
            N_i = train_features.shape[1]

            # create network with optimal hyper parameters
            network = OneLayerNeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)
            # train
            losse_df = OneLayerNeuralNetwork.get_loss_per_epoch(network, iterations, train_features,
                                                                train_targets, val_features, val_targets)
        pd.testing.assert_frame_equal(losse_df_to_match, losse_df)


if __name__=='__main__':
    unittest.main()