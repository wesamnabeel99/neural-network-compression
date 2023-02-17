import numpy as np

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


class DeepNeuralNetworkLayers:
    def __init__(self, input_neurons, hidden_neurons, output_neurons):
        self.input_neurons = input_neurons
        self.hidden_neurons = hidden_neurons
        self.output_neurons = output_neurons
        self.hidden_weights = np.random.uniform(-1, 1, size=(hidden_neurons, input_neurons))
        self.output_weights = np.random.uniform(-1, 1, size=(output_neurons, hidden_neurons))

    def train(self, epoch_size, alpha, images, labels):
        one_hot_encoding = np.eye(self.output_neurons)[labels.astype(int)]
        accuracy_train = []

        for epoch in range(epoch_size):
            accum_acc = 0
            for i in range(images.shape[0]):
                hidden_logit = np.dot(images[i, :], self.hidden_weights.T)
                hidden_output = sigmoid(hidden_logit)

                output_logit = np.dot(hidden_output, self.output_weights.T)
                final_output = sigmoid(output_logit)

                # Training part
                error = final_output - one_hot_encoding[i]
                self.output_weights -= alpha * np.outer(error, hidden_output)

                # network evaluation
                winning_class = np.argmax(final_output)

                # compare the  winning class with the ground truth
                evaluation = 1.0 * (winning_class == labels[i])
                accum_acc += evaluation

            # final result
            accuracy_train.append((accum_acc / len(labels)) * 100)
        return accuracy_train

    def test(self, images, labels):
        accuracy_test = []
        accum_acc = 0
        for i in range(images.shape[0]):
            hidden_logit = np.dot(images[i, :], self.hidden_weights.T)
            hidden_output = sigmoid(hidden_logit)

            output_logit = np.dot(hidden_output, self.output_weights.T)
            final_output = sigmoid(output_logit)

            winning_class = np.argmax(final_output)
            # compare the  winning class with the ground truth
            evaluation = 1.0 * (winning_class == labels[i])
            accum_acc += evaluation

        accuracy_test.append((accum_acc / len(labels) * 100))
        return accuracy_test

    def convolve(self, image, kernel):
        output = np.zeros([100, 676], dtype=float, order='C')
        image_2d = image[:, :].reshape(28, -1)
        for i in range(image.shape[0]):
            for j in range(26):
                for k in range(26):
                    for m in range(3):
                        for n in range(3):
                            output[j][k] += image_2d[j + m][k + n] * kernel[m][n]

        return output
