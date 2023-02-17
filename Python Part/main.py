"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
import numpy as np
from utils import constants
from lib.cnn import cnn_layers
from lib.dnn_layers import dnn_layers
from utils.read_mnist_data import read_mnist_data

# extracting the image
images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, constants.TRAIN_SAMPLE_SIZE)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, constants.TEST_SAMPLE_SIZE)

# Defining the hyperparameter
input_neurons = images_train.shape[1]
hidden_neurons = 100
output_neurons = 10

cnn = cnn_layers()
# Defining the convolutional parameters
image_size = images_train[0].shape[0]
n_kernels = 3
n_channels = 1

# Setting the shapes and sizes for convolution operation
output_shape = (image_size - n_kernels + 1, image_size - n_kernels + 1)
kernel_shape = np.random.uniform(-1, 1, size=(n_kernels, n_kernels))
stride_array = np.random.uniform(-1, 1, size=output_shape)

# For the training
images_filtered_training = cnn.convolve(images_train, kernel_shape)
final_images_training = cnn.pool(images_filtered_training)

# For the testing
images_filtered_test = cnn.convolve(images_test, kernel_shape)
final_images_test = cnn.pool(images_filtered_test)

# TODO: pass the image size (should be 676 but something is wrong), try to debug and see what's wrong
# final_images_training.flatten().shape[0] output is 1954609!! it should be 676
dnn = dnn_layers(final_images_training.flatten().shape[0], hidden_neurons, output_neurons)

# TODO: flatten the images and pass them to the neural network
dnn.evaluate_model(
    epoch_size=100, alpha=0.1, images_test=final_images_test.flatten(),
    labels_test=labels_test, images_train=final_images_training.flatten(), labels_train=labels_train
)