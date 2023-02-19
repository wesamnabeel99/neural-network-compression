"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
import winsound

from utils import constants
from utils.read_mnist_data import read_mnist_data
from models import *

images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, constants.TRAIN_SAMPLE_SIZE)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, constants.TEST_SAMPLE_SIZE)

model_one(images_train,labels_train,images_test,labels_test)
winsound.Beep(440, 1000)

model_two(images_train,labels_train,images_test,labels_test)
winsound.Beep(440, 1000)

model_three(images_train,labels_train,images_test,labels_test)
winsound.Beep(440, 1000)

model_four(images_train,labels_train,images_test,labels_test)
winsound.Beep(440, 1000)

model_five(images_test,labels_test)
winsound.Beep(440, 1000)