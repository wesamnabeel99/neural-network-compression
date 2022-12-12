""""
Author: Abdullah Haware Theeb
Data: November 2/11/2022
Descripation: This is to implement multi neural network

"""
from scipy import signal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Define the sigmoid functions
def sigmoid(logit):
    return 1.0 / (1 + numpy.exp(-logit))


# extracting the image
data_train = pd.read_csv(r"C:\Users\MSI GF63\Downloads\mnist_test.csv ", header=0, nrows=1).values
images = data_train[:, 1:]
labels = data_train[:, 0]

# Taking sample image
sample = images[[0], :].reshape(28, 28)
plt.imshow(sample)
plt.savefig("C:/Users/MSI GF63/Downloads/ NNP")


# Defining the varaibles
input_neuron = images.shape[1]
hidden_neuron = 100
output_neurons = 10
images_to_be_trained = images.shape[0]
input_height = sample.shape[0]
kernel_size = 3
depth = 3
input_depth = 3
output_shape = (depth, input_height - kernel_size + 1, input_height - kernel_size + 1)
kernel_shape = np.random.uniform(-1, 1, size=(depth, input_depth, kernel_size, kernel_size))
biases_shape = np.random.uniform(-1, 1, size=(output_shape))
output = np.copy(biases_shape)
# The output from convolution the image with the filter.
for i in range(input_height - 2):
    for j in range(50):
        output += signal.correlate2d(images[i], kernel_shape[i, j], "valid")
        print(output)