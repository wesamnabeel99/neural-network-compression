"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
import winsound

from utils import constants
from utils.read_mnist_data import read_mnist_data
from network_models import *

sample_size = 2000
train_sample_size = 0.7 * sample_size
test_sample_size = 0.3 * sample_size

images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, train_sample_size)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, test_sample_size)

network_models = network_models(
    images_train=images_train, labels_train=labels_train, images_test=images_test, labels_test=labels_test,
    n_kernels=3,kernel_size=3, epoch=10, alpha=0.1, hidden=100
)

network_models.model_one()

network_models.model_six()
winsound.Beep(440, 1000)

winsound.Beep(800, 500)