import numpy as np
from lib.sigmoid import sigmoid
from utils.report_generator import generate_report

class fnn_layers:
    def __init__(self, input_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.output_neurons = output_neurons

        np.random.seed(1000)

        print("\n\n")
        print("built a neural network with %d input, %d output" % (
            self.input_neurons, self.output_neurons))
        print("\n\n")
        self.__output_weights = np.random.uniform(-1, 1, size=(
            self.output_neurons, input_neurons))

    def evaluate_model(self, epoch_size, alpha, images_train, labels_train, images_test, labels_test, model_name):
        accuracy_train = []
        accuracy_test = []
        for epoch in range(epoch_size):
            print(f"================----- epoch: {epoch} -----================")
            accuracy_train.append(self.backward_without_hidden(alpha=alpha, labels=labels_train, images=images_train))
            accuracy_test.append(self.forward_without_hidden(images=images_test, labels=labels_test))

        generate_report(
            accuracy_train=accuracy_train, accuracy_test=accuracy_test, epoch_size=epoch_size,
            training_sample_size=images_train.shape[0], testing_sample_size=images_test.shape[0], alpha=alpha,
            input=self.input_neurons, output=self.output_neurons, model_name=model_name
        )

    def forward_without_hidden(self, images, labels):
        print("forward-------------->>>>>>")
        accum_acc = 0
        for i in range(images.shape[0]):
            final_output = self.calculate_neuron_output(images[i, :], self.__output_weights)

            winning_class = np.argmax(final_output)

            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        print("###forward finished###")
        return accum_acc / len(labels) * 100

    def backward_without_hidden(self, images, labels, alpha):
        one_hot_encoding = np.eye(self.output_neurons)[labels.astype(int)]

        print("backward-------------->>>>>>")
        accum_acc = 0
        for i in range(images.shape[0]):
            final_output = self.calculate_neuron_output(images[i, :], self.__output_weights)
            # Training part
            error = final_output - one_hot_encoding[i]
            self.__output_weights -= alpha * np.outer(error, images[i, :])

            winning_class = np.argmax(final_output)

            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        print("###backward finished###")
        return accum_acc / len(labels) * 100

    def test_neural_network(self, image):
        return self.calculate_neuron_output(image, self.__output_weights)
    def calculate_neuron_output(self, neuron, weight):
        logit = np.dot(neuron, weight.T)
        return sigmoid(logit)