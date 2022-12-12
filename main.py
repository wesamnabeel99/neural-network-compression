"""
Author: Abdullah Haware Theeb
Data: November 2, 2022
Descripation: This is to implement multi neural network
"""
from lib.sigmoid import sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from numpy import array


def convolve(image, kernal, stride_array):
    output = np.empty_like(images)
    output = np.copy(stride_array)
    for i in range(image_size - 2):
        output[:, i] += signal.convolve2d(images[:, i], kernel_shape[:, i], boundary="symm")
    return output


# extracting the image
data_train = pd.read_csv(r"C:\Users\MSI GF63\Downloads\mnist_train.csv ", header=0, nrows=100).values
images = data_train[:, 1:]
labels = data_train[:, 0]

# Taking sample image
sample = images[[0], :].reshape(28, 28)
plt.imshow(sample)
plt.savefig("C:/Users/MSI GF63/Downloads/ NNP")
one_hot_encoding = pd.get_dummies(labels)

# Defing the varaibles
input_nuerons = images.shape[1]
hidden_nuerons = 100
output_nuerons = 10
images_to_be_trained = images.shape[0]
alpha = 0.1
image_size = sample.shape[0]
kernel_size = 3
depth = 1

# Setting the shapes and sizes for convolution operation
output_shape = (image_size - kernel_size + 1, image_size - kernel_size + 1)
kernel_shape = np.random.uniform(-1, 1, size=(kernel_size, kernel_size))
stride_array = np.random.uniform(-1, 1, size=output_shape)
images_filtered = convolve(images, kernel_shape, stride_array)
# The output from convolution the image with the filter.


# Giving the layers there weights
hidden_weights = np.random.uniform(-1, 1, size=(hidden_nuerons, input_nuerons))
output_weights = np.random.uniform(-1, 1, size=(output_nuerons, hidden_nuerons))

acc_tr = []
accum_acc = 0

for i in range(images_to_be_trained):
    hidden_logit = np.dot(images_filtered[[i], :], hidden_weights.T)
    hidden_output = sigmoid(hidden_logit)
    output_logit = np.dot(hidden_logit, output_weights.T)
    output_output = sigmoid(output_logit)

    # To be use later
    # error = np.sum(output_output - one_hot_encoding) ** 2
    # cost = 0.5 * error
    # To be use later
    # cost1 iS J * H
    # cost1 = np.dot(cost, hidden_logit).T
    # x = (output_output - one_hot_encoding)

    # Traning part
    cost_function = output_output - one_hot_encoding
    new_weight = output_weights - (alpha * cost_function.T)
    new_hidden_logit = np.dot(images[[i], :], new_weight.T)
    new_hidden_output = sigmoid(new_hidden_logit)
    new_output_logit = np.dot(new_hidden_logit, new_weight.T)
    new_output = sigmoid(new_output_logit)
    print("the new output is :", new_output)

    # network evaluation
    winning_class = np.argmax(new_output)

    # compare the  winning class with the gruond truth
    evaluation = (winning_class == labels[i])
    accum_acc += evaluation

# final result
acc_tr.append((accum_acc / images_to_be_trained) * 100)
print("final tranning accuracy = %.3f" % acc_tr[-1])
# With plot
# Tesing
data_train = pd.read_csv(r"C:\Users\MSI GF63\Downloads\mnist_test.csv ", header=0, nrows=100).values
images = data_train[:, 1:]
for j in range(100):
    hidden_logit = np.dot(images_filtered[[j], :], new_weight.T)
    hidden_output = sigmoid(hidden_logit)
    output_logit = (hidden_output, new_weight.T)
    output_output = sigmoid(output_logit)
    winning_class = np.argmax(output_output)

    # compare the  winning class with the gruond truth
    evaluation = (winning_class == labels[i])
    accum_acc += evaluation

acc_tr.append((accum_acc / images_to_be_trained) * 100)
print("final tranning accuracy = %.3f" % acc_tr[-1])
#  With plot
