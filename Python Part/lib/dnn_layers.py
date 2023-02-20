import numpy as np

from lib.sigmoid import sigmoid
from utils.report_generator import generate_report


class dnn_layers:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        print("\n\n")
        print("built a neural network with %d input, %d hidden, %d output" % (self.input_neurons, self.hidden_neurons, self.output_neurons))
        print("\n\n")
        self.__hidden_weights = np.random.uniform(-1, 1, size=(hidden_neurons, input_neurons))
        self.__output_weights = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons))

    def evaluate_model(self, epoch_size, alpha, images_train, labels_train, images_test, labels_test,model_name):
        accuracy_train = []
        accuracy_test = []
        for epoch in range(epoch_size):
            print(f"================----- epoch: {epoch} -----================")
            accuracy_train.append(self.backward(alpha=alpha, labels=labels_train, images=images_train))
            accuracy_test.append(self.forward(images=images_test, labels=labels_test))

        generate_report(
            accuracy_train=accuracy_train, accuracy_test=accuracy_test, epoch_size=epoch_size,
            training_sample_size=images_train.shape[0], testing_sample_size=images_test.shape[0], alpha=alpha,
            input=self.input_neurons, hidden=self.hidden_neurons, output=self.output_neurons,model_name = model_name
        )
        print(f"training is done and report is generated")

    def backward(self, alpha, images, labels):
        print("<<<<<-----------backward")
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

        print(f"training accuracy: {(accum_acc / len(labels)) * 100}")
        print("###backward finished###")

        return (accum_acc / len(labels)) * 100

    def forward(self, images, labels):
        print("forward-------------->>>>>>")
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

        print(f"test accuracy: {(accum_acc / len(labels)) * 100}")
        print("###forward finished###")
        return accum_acc / len(labels) * 100
