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
        accuracy_train = []
        accuracy_test = []
        for epoch in range(epoch_size):
            accuracy_train.append(self.__train(alpha=alpha, labels=labels_train, images=images_train))
            accuracy_test.append(self.__test(images=images_test, labels=labels_test))

        generate_report(
            accuracy_train=accuracy_train, accuracy_test=accuracy_test, epoch_size=epoch_size,
            training_sample_size=images_train.shape[0], testing_sample_size=images_test.shape[0], alpha=alpha
        )

    def __train(self, alpha, images_train, labels, kernel_shape):

        one_hot_encoding = np.eye(self.output_neurons)[labels.astype(int)]
        accum_acc = 0
        for i in range(images_train.shape[0]):
            images_2d = images_train[i].reshape(28, 28)
            images_filtered =cnn.convolve(images_2d, kernel_shape)
            final_image = cnn.pool(images_filtered)
            hidden_logit = np.dot(final_image.flatten(), self.__hidden_weights.T)
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

    def __test(self, images_test, labels, kernel_shape):
        accum_acc = 0
        for i in range(images_test.shape[0]):
            images_2d = images_test[i].reshape(28, 28)
            images_filtered = cnn.convolve(images_2d, kernel_shape)
            final_image = cnn.pool(images_filtered)
            hidden_logit = np.dot(final_image.flatten(), self.__hidden_weights.T)
            hidden_output = sigmoid(hidden_logit)

            output_logit = np.dot(hidden_output, self.__output_weights.T)
            final_output = sigmoid(output_logit)

            winning_class = np.argmax(final_output)
            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        return accum_acc / len(labels) * 100
