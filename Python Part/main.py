"""
Author: Abdullah Haware Theeb
Data: November 2, 2022
Descripation: This is to implement multi neural network

"""
import numpy as np

from lib.cnn import cnn
from lib.dnn_layers import dnn_layers
from utils.read_mnist_data import read_mnist_data

# extracting the image
images_train, labels_train = read_mnist_data("dataset/mnist_train.csv", 1000)
images_test, labels_test = read_mnist_data("dataset/mnist_test.csv", 100)

# Defining the hyperparameter
input_neurons = images_train.shape[1]
hidden_neurons = 100
output_neurons = 10

cnn = cnn()
dnn = dnn_layers(input_neurons, hidden_neurons, output_neurons)

one_hot_encoding = np.eye(10)[labels_train.astype(int)]

# Defining the convolutional parameters
image_size = images_train[0].shape[0]
n_kernels = 3
n_channels = 1

# Setting the shapes and sizes for convolution operation
output_shape = (image_size - n_kernels + 1, image_size - n_kernels + 1)
kernel_shape = np.random.uniform(-1, 1, size=(n_kernels, n_kernels))
stride_array = np.random.uniform(-1, 1, size=output_shape)
images_filtered = cnn.convolve(images_train, kernel_shape)

dnn.evaluate_model(
    epoch_size=100, alpha=0.1, images_test=images_test,
    labels_test=labels_test, images_train=images_train, labels_train=labels_train
)