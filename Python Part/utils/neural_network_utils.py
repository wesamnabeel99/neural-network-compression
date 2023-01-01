import numpy
from lib.sigmoid import sigmoid

# Change the input no
def test_neural_network(input_neurons, hidden_weights, output_weights):
    hidden_output = calculate_neuron_output(input_neurons, hidden_weights)
    return calculate_neuron_output(hidden_output, output_weights)


def calculate_neuron_output(neuron, weight):
    logit = numpy.dot(neuron,weight.T)
    return sigmoid(logit)
