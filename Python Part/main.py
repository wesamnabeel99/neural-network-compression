"""
Author: Abdullah Haware Theeb
Data: November 2, 2022
Descripation: This is to implement multi neural network

"""
import matplotlib.pyplot as plt
import numpy as np

from lib.deepneuralnetworklayers import DeepNeuralNetworkLayers
from utils.read_mnist_data import read_mnist_data

# extracting the image
images_train, labels_train = read_mnist_data("dataset/mnist_train.csv", 1000)
images_test, labels_test = read_mnist_data("dataset/mnist_test.csv", 100)

# Defining the hyperparameter
input_neurons = images_train.shape[1]
hidden_neurons = 100
output_neurons = 10
alpha = 0.5
epoch_size = 50

dnn = DeepNeuralNetworkLayers(input_neurons, hidden_neurons, output_neurons)

one_hot_encoding = np.eye(10)[labels_train.astype(int)]

# Defining the convolutional parameters
image_size = images_train[0].shape[0]
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

accuracy_train = dnn.train(epoch_size=epoch_size, alpha=alpha, images=images_train, labels=labels_train)

accuracy_test = dnn.test(images=images_test, labels=labels_test)

print("final training accuracy = %.3f" % accuracy_train[-1])
print("final test accuracy = %.3f" % accuracy_test[-1])

plt.figure(2)
plot = plt.plot(accuracy_train)
plt.plot(accuracy_test)
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()
