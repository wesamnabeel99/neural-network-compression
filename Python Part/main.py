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

cnn = cnn_layers()

# Defining the convolutional parameters
n_kernels = 3

# Setting the shapes and sizes for convolution operation
kernel_shape = np.random.uniform(0, 1, size=(n_kernels, n_kernels))

# normalize the kernel
kernel_sum = np.sum(kernel_shape)
if kernel_sum != 0:
    kernel_shape = kernel_shape / kernel_sum


for i in range(images_train.shape[0]):
    images = images_train[i].reshape(28, 28)
    images = cnn.convolve(images, kernel_shape)
    images = cnn.pool(images)
    images = images.flatten()

for i in range(images_test.shape[0]):
    images = images_test[i].reshape(28, 28)
    images = cnn.convolve(images, kernel_shape)
    images = cnn.pool(images)
    images = images.flatten()


# Defining the hyperparameter
input_neurons = images_train.shape[1]
hidden_neurons = 100
output_neurons = 10

dnn = dnn_layers(input_neurons, hidden_neurons, output_neurons)

dnn.evaluate_model(
    epoch_size=100, alpha=0.1, images_test=images_test,
    labels_test=labels_test, images_train=images_train, labels_train=labels_train
)
print("hi")