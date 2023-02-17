import numpy as np

from lib.sigmoid import sigmoid
from utils.report_generator import generate_report

class dnn_layers:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.__hidden_weights = np.random.uniform(-1, 1, size=(hidden_neurons, input_neurons))
        self.__output_weights = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons))

    def evaluate_model(self, epoch_size, alpha, images_train, labels_train, images_test, labels_test):
        train_accuracy = []
        test_accuracy = []
        for epoch in range(epoch_size):
            train_accuracy.append(self.__train(alpha=alpha, labels=labels_train, images=images_train))
            test_accuracy.append(self.__test(images=images_test, labels=labels_test))

        generate_report(
            accuracy_train=train_accuracy, accuracy_test=test_accuracy, epoch_size=epoch_size,
            training_sample_size=images_train.shape[0],
            testing_sample_size=images_test.shape[0], alpha=alpha
        )

    def __train(self, alpha, images, labels):
        one_hot_encoding = np.eye(self.output_neurons)[labels.astype(int)]

        accum_acc = 0
        for i in range(images.shape[0]):
            hidden_logit = np.dot(images[i, :], self.__hidden_weights.T)
            hidden_output = sigmoid(hidden_logit)

            output_logit = np.dot(hidden_output, self.__output_weights.T)
            final_output = sigmoid(output_logit)

            # Training part
            error = final_output - one_hot_encoding[i]
            self.__output_weights -= alpha * np.outer(error, hidden_output)

            # network evaluation
            winning_class = np.argmax(final_output)

            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        return (accum_acc / len(labels)) * 100

    def __test(self, images, labels):
        accum_acc = 0
        for i in range(images.shape[0]):
            hidden_logit = np.dot(images[i, :], self.__hidden_weights.T)
            hidden_output = sigmoid(hidden_logit)

            output_logit = np.dot(hidden_output, self.__output_weights.T)
            final_output = sigmoid(output_logit)

            winning_class = np.argmax(final_output)
            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        return accum_acc / len(labels) * 100
