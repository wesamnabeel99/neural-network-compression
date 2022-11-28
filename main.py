import matplotlib.pyplot as matplot
import numpy
import pandas

from utils.neural_network_utils import test_neural_network

test_file_path = "dataset/mnist_test.csv"
train_file_path = "dataset/mnist_train.csv"

data_train = pandas.read_csv(train_file_path, header=None, nrows=100).values

images, ground_truth = data_train[:, 1:], data_train[:, 0]

sample_image = images[[0], :].reshape(28, 28)

matplot.imshow(sample_image)
matplot.savefig("temp/sample.png")

LABELS_SHAPE = 0
PIXELS_SHAPE = 1

input_neuron = images.shape[PIXELS_SHAPE]

labels = images.shape[LABELS_SHAPE]
hidden_neurons = 100
output_neurons = 10

hidden_weights = numpy.random.uniform(-1, 1, size=(hidden_neurons, input_neuron))
output_weights = numpy.random.uniform(-1, 1, size=(output_neurons, hidden_neurons))

evaluation = 0

alpha = 0.1
one_hot_encoding = pandas.get_dummies(ground_truth)
error = 0
for i in range(labels):
    input_image = images[[i], :]

    output_output = test_neural_network(
    input_neurons=input_image,
    hidden_weights=hidden_weights,
    output_weights=output_weights,
    )

    error = numpy.sum((output_output - one_hot_encoding)**2)

    cost = error / 2

    cost_function = numpy.multiply(numpy.subtract(output_output,one_hot_encoding),hidden_weights)

    hidden_weights = numpy.subtract(hidden_weights, cost_function * alpha)

    winning_class = numpy.argmax(output_output)
    evaluation += winning_class == ground_truth[i]

accuracy = evaluation / labels * 100
print(evaluation / labels * 100)
print(error)
