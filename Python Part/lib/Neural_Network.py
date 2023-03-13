import numpy as np
from lib.dnn_layers import dnn_layers

class Neural_Network:
    def __init__(self, input_neurons, output_neurons, hidden_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons
        self.hidden_neurons = hidden_neurons
        np.random.seed(1000)

        print("\n\n")
        print("built a neural network with %d input, %d hidden, %d output" % (
            self.input_neurons, self.output_neurons, self.hidden_neurons))
        print("\n\n")
        self.__output_weights = np.random.uniform(-1, 1, size=(
            self.output_neurons, input_neurons))

    def evaluate_model(self, epoch_size, alpha, images_train, labels_train, images_test, labels_test, model_name):
        accuracy_train = []
        accuracy_test = []
        for epoch in range(epoch_size):
            print(f"================----- epoch: {epoch} -----================")
            accuracy_train.append(dnn_layers.backward(alpha=alpha, labels=labels_train, images=images_train))
            accuracy_test.append(dnn_layers.forward(images=images_test, labels=labels_test))