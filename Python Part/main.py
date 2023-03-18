"""
Author: Abdullah H. Theeb, Wesam N. Shawqi
Date: November 2, 2022
Description: multi neural network implementation
"""
# timeit package for measuring the run speed

import winsound
import timeit
import time
from utils import constants
from utils.read_mnist_data import read_mnist_data
from network_models import *

sample_size = 5000
train_sample_size = 0.7 * sample_size
test_sample_size = 0.3 * sample_size

images_train, labels_train = read_mnist_data(constants.MNIST_TRAIN_FILEPATH, train_sample_size)
images_test, labels_test = read_mnist_data(constants.MNIST_TEST_FILEPATH, test_sample_size)


user_input_start = input("start running from model number:")
user_input_end = input("to model number:")

network_models = network_models(
    images_train=images_train, labels_train=labels_train, images_test=images_test, labels_test=labels_test,
    n_kernels=3, kernel_size=3, epoch=40, alpha=0.1, hidden=100
)

for i in range(int(user_input_start),int(user_input_end)):
    if i==1:
        network_models.model_one()
    elif i==2:
        network_models.model_two()
    elif i==3:
        network_models.model_three()
    elif i==4:
        network_models.model_four()
    elif i==5:
        network_models.model_five()
    elif i==6:
        network_models.model_six()
    elif i==7:
        network_models.model_seven()
    else:
        print("ther's no such model as model %d"%(i))

winsound.Beep(800, 500)
