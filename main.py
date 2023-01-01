"""
Author: Abdullah Haware Theeb
Data: November 2, 2022
Descripation: This is to implement multi neural network

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.dnn_layers import dnn_layers
from lib.sigmoid import sigmoid

dnn = dnn_layers()

# extracting the image
data_train = pd.read_csv("dataset/mnist_train.csv", header=None, nrows=1500).values
data_test = pd.read_csv("dataset/mnist_test.csv", header=None, nrows=100).values
images_train, labels_train = data_train[:, 1:], data_train[:, 0]
images_test, labels_test = data_test[:, 1:], data_test[:, 0]

# Normalize the image features
images_train = images_train / 255.0
images_test = images_test / 255.0

# Taking sample_train image for training
sample_train = images_train[[0], :].reshape(28, 28)
plt.imshow(sample_train)
plt.savefig("temp/train_image.png")

# Taking sample_train image for testing
sample_test = images_test[[0], :].reshape(28, 28)
plt.imshow(sample_test)
plt.savefig("temp/test_image.png")

one_hot_encoding = np.eye(10)[labels_train.astype(int)]

# Defining the convolutional parameters
image_size = sample_train.shape[0]
n_kernels = 3
n_channels = 1

# Setting the shapes and sizes for convolution operation
output_shape = (image_size - n_kernels + 1, image_size - n_kernels + 1)
kernel_shape = np.random.uniform(-1, 1, size=(n_kernels, n_kernels))
stride_array = np.random.uniform(-1, 1, size=output_shape)
images_filtered = dnn.convolve(images_train, kernel_shape)
sample3 = images_filtered[[9], :].reshape(26, 26)
plt.figure(num=1)
plt.imshow(sample3)
plt.savefig('temp/meow.png')

# Defining the hyperparameter
input_neurons = images_train.shape[1]
hidden_neurons = 100
output_neurons = 10
images_to_be_trained = images_train.shape[0]
images_to_be_tested = images_test.shape[0]
alpha = 0.5
epoch_size = 50

# Giving the layers there weights
hidden_weights = np.random.uniform(-1, 1, size=(hidden_neurons, input_neurons))
output_weights = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons))

acc_tr = []
acc_tes = []
for epoch in range(epoch_size):
    accum_acc = 0
    for i in range(images_to_be_trained):
        hidden_logit = np.dot(images_train[i, :], hidden_weights.T)
        hidden_output = sigmoid(hidden_logit)

        output_logit = np.dot(hidden_output, output_weights.T)
        final_output = sigmoid(output_logit)

        # Training part
        error = final_output - one_hot_encoding[i]
        output_weights -= alpha * np.outer(error, hidden_output)
        # network evaluation
        winning_class = np.argmax(final_output)

        # compare the  winning class with the ground truth
        evaluation = 1.0 * (winning_class == labels_train[i])
        accum_acc += evaluation

    # final result
    acc_tr.append((accum_acc / len(labels_train)) * 100)

    accum_acc = 0
    for i in range(images_to_be_tested):
        hidden_logit = np.dot(images_test[i, :], hidden_weights.T)
        hidden_output = sigmoid(hidden_logit)

        output_logit = np.dot(hidden_output, output_weights.T)
        final_output = sigmoid(output_logit)

        winning_class = np.argmax(final_output)
        # compare the  winning class with the ground truth
        evaluation = 1.0 * (winning_class == labels_test[i])
        accum_acc += evaluation

    acc_tes.append((accum_acc / len(labels_test) * 100))

# Testing


print("final training accuracy = %.3f" % acc_tr[-1])
print("final training accuracy = %.3f" % acc_tes[-1])

plt.figure(2)
plot = plt.plot(acc_tr)
plt.plot(acc_tes)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()