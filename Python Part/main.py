"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
import winsound

from lib.cnn import cnn_layers
from lib.dnn_layers import dnn_layers
from utils import constants
from utils.image_helper import *
from utils.read_mnist_data import read_mnist_data

# extracting the image
images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, constants.TRAIN_SAMPLE_SIZE)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, constants.TEST_SAMPLE_SIZE)

cnn = cnn_layers(n_kernels=3)

images_train = reshape_all_images(images_train)
images_test = reshape_all_images(images_test)

images_train = cnn.convolve(images_train)
images_train = cnn.pool(images_train)
images_train = flatten_all_images(images_train)

images_test = cnn.convolve(images_test)
images_test = cnn.pool(images_test)
images_test = flatten_all_images(images_test)

dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=100, output_neurons=10)

dnn.evaluate_model(
    epoch_size=50, alpha=0.1, images_test=images_test,
    labels_test=labels_test, images_train=images_train, labels_train=labels_train
)
winsound.Beep(440, 1000)
