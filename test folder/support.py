import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def convolve(image, kernel):
    output = np.zeros([26, 26], dtype=float, order='C')
    image_2d = image[:, :].reshape(28, -1)
    for i in range(image.shape[0]):
        for j in range(26):
            for k in range(26):
                for m in range(3):
                    for n in range(3):
                        output[j][k] += image_2d[j + m][k + n] * kernel[m][n]
    return output

# Load data
data_train =  pd.read_csv(r"C:\Users\MSI GF63\Downloads\mnist_train.csv ", header=0, nrows=100).values
data_test = pd.read_csv(r"C:\Users\MSI GF63\Downloads\mnist_test.csv ", header=0, nrows=100).values

# Split data into images and labels
images, labels = data_train[:, 1:], data_train[:, 0]
images2, labels2 = data_test[:, 1:], data_test[:, 0]

# One-hot encode labels
one_hot_encoding = np.eye(10)[labels.astype(int)]

# Define convolutional parameters
image_size = 28
n_kernels = 3
n_channels = 1

# Set shapes and sizes for convolution operation
output_shape = (image_size - n_kernels + 1, image_size - n_kernels + 1)
kernel_shape = np.random.uniform(-1, 1, size=(n_kernels, n_kernels))
stride_array = np.random.uniform(-1, 1, size=output_shape)
images_filtered = convolve(images, kernel_shape)
images_flat = images_filtered.reshape(images_filtered.shape[0], -1)

# Define hyperparameters
input_neurons = images_flat.shape[1]
hidden_neurons1 = 256
hidden_neurons2 = 128
output_neurons = 10
images_to_be_trained = images.shape[0]
images_to_be_tested = images2.shape[0]
alpha = 0.01

# Initialize weights and biases
hidden_weights1 = np.random.uniform(-1, 1, size=(hidden_neurons1, input_neurons))
hidden_bias1 = np.zeros(hidden_neurons1)
hidden_weights2 = np.random.uniform(-1, 1, size=(hidden_neurons2, hidden_neurons1))
hidden_bias2 = np.zeros(hidden_neurons2)
output_weights = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons2))
output_bias = np.zeros(output_neurons)
acc_tr = []
for epoch in range(100):
    accum_acc = 0
    for i in range(images_to_be_trained):
        hidden_logit1 = np.dot(images_flat[i], hidden_weights1.T) + hidden_bias1
        hidden_output1 = relu(hidden_logit1)
        hidden_logit2 = np.dot(hidden_output1, hidden_weights2.T) + hidden_bias2
        hidden_output2 = relu(hidden_logit2)
        output_logit = np.dot(hidden_output2, output_weights.T) + output_bias
        final_output = sigmoid(output_logit)

        # Calculate error
        error = final_output - one_hot_encoding[i]
        output_weights -= alpha * np.dot(error, hidden_output2)
        hidden_weights2 -= alpha * np.dot(error.T, hidden_output1)
        hidden_weights1 -= alpha * np.dot(error.T, images_flat[i])
        output_bias -= alpha * error
        hidden_bias2 -= alpha * np.dot(error.T, hidden_output1)
        hidden_bias1 -= alpha * np.dot(error.T, images_flat[i])

        # Determine winning class and compare to ground truth
        winning_class = np.argmax(final_output)
        ground_truth = np.argmax(one_hot_encoding[i])
        evaluation = (winning_class == ground_truth)
        accum_acc += evaluation

    # Calculate and store final training accuracy for this epoch
    acc_tr.append((accum_acc / images_to_be_trained) * 100)

# Plot training accuracy
plt.plot(acc_tr)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()