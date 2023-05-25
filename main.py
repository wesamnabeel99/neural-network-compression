"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""

import winsound
from utils import constants
from utils.read_mnist_data import read_mnist_data
from network_models import *

sample_size = 5000
train_sample_size = 0.7 * sample_size
test_sample_size = 0.3 * sample_size

images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, train_sample_size)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, test_sample_size)

kernel_size= int(input("Please enter the kernel size:"))
number_of_kernels = int(input("Please enter the number of kernels"))
alpha = float(input("Learning Rate:"))
epochs = int(input("Epochs:"))
hidden = int(input("Hidden:"))


network_models = network_models(
    images_train=images_train, labels_train=labels_train, images_test=images_test, labels_test=labels_test,
    n_kernels=number_of_kernels, kernel_size=kernel_size, epoch=epochs, alpha=alpha, hidden=100
)

while True:
    model_number = input("Choose a model number to run (1-7), or write anything else to stop: ")
    if not model_number.isdigit():
        break
    model_number = int(model_number)
    if 1 <= model_number <= 7:
        getattr(network_models, f"model_{model_number}")()
        winsound.Beep(800, 500)
    else:
        print(f"Model {model_number} does not exist.")