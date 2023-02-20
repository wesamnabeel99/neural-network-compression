"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
import winsound

from utils import constants
from utils.read_mnist_data import read_mnist_data
from network_models import *

images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, constants.TRAIN_SAMPLE_SIZE)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, constants.TEST_SAMPLE_SIZE)

network_models = network_models(
    images_train=images_train, labels_train=labels_train, images_test=images_test, labels_test=labels_test,
    n_kernels=3, epoch=10, alpha=0.1, hidden=100
)

network_models.model_one()
winsound.Beep(440, 1000)

network_models.model_two()
winsound.Beep(440, 1000)

network_models.model_three()
winsound.Beep(440, 1000)

network_models.model_four()
winsound.Beep(440, 1000)

network_models.model_five()
winsound.Beep(440, 1000)

winsound.Beep(800, 500)
