from lib.cnn import cnn_layers
from lib.dnn_layers import dnn_layers
from utils.image_helper import *
from lib.fnn import fnn_layers


class network_models:
    def __init__(self, images_train, labels_train, images_test, labels_test, n_kernels, kernel_size, epoch, alpha,
                 hidden):
        self.images_train = images_train
        self.labels_train = labels_train
        self.images_test = images_test
        self.labels_test = labels_test
        self.cnn = cnn_layers(n_kernels=n_kernels, kernel_size=kernel_size)
        self.epoch = epoch
        self.alpha = alpha
        self.hidden = hidden
        self.n_kernels = n_kernels

    def model_1(self):
        model_name = "cnn_pooling_fully_connected"

        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")
        images_train = reshape_all_images(self.images_train)
        images_test = reshape_all_images(self.images_test)

        images_train = self.cnn.convolve(images_train)
        images_train = self.cnn.pool(images_train)
        images_train = flatten_all_images(images_train)

        images_test = self.cnn.convolve(images_test)
        images_test = self.cnn.pool(images_test)
        images_test = flatten_all_images(images_test)

        dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        dnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=images_test,
            labels_test=self.labels_test, images_train=images_train, labels_train=self.labels_train,
            model_name=model_name
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_2(self):
        model_name = "cnn_fully_connected"
        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")

        images_train = reshape_all_images(self.images_train)
        images_test = reshape_all_images(self.images_test)

        images_train = self.cnn.convolve(images_train)
        images_train = flatten_all_images(images_train)

        images_test = self.cnn.convolve(images_test)
        images_test = flatten_all_images(images_test)

        dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        dnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=images_test,
            labels_test=self.labels_test, images_train=images_train, labels_train=self.labels_train,
            model_name=model_name
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_3(self):
        model_name = "fully_connected"
        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")
        dnn = dnn_layers(input_neurons=self.images_train.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        dnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=self.images_test,
            labels_test=self.labels_test, images_train=self.images_train, labels_train=self.labels_train,
            model_name="dnn"
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_4(self):
        model_name = "pooling_fully_connected"
        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")

        images_train = reshape_all_images(self.images_train)
        images_test = reshape_all_images(self.images_test)

        images_train = self.cnn.pool(images_train)
        images_train = flatten_all_images(images_train)

        images_test = self.cnn.pool(images_test)
        images_test = flatten_all_images(images_test)

        dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        dnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=images_test,
            labels_test=self.labels_test, images_train=images_train, labels_train=self.labels_train,
            model_name=model_name
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_5(self):
        model_name = "cnn_fully_connected_forward"
        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")

        images_test = reshape_all_images(self.images_test)

        images_test = self.cnn.convolve(images_test)
        images_test = flatten_all_images(images_test)

        dnn = dnn_layers(input_neurons=images_test.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        accuracy = dnn.forward(images_test, self.labels_test)
        print(f"model {model_name} accuracy: {accuracy}")
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_6(self):
        model_name = "cnn_multi_kernel_fully_connected"

        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")
        images_train = reshape_all_images(self.images_train)
        images_test = reshape_all_images(self.images_test)

        images_train = self.cnn.convolve_multiple_kernels(images_train)
        images_train = self.cnn.pool(images_train)
        images_train = flatten_all_images(images_train)

        images_test = self.cnn.convolve_multiple_kernels(images_test)
        images_test = self.cnn.pool(images_test)
        images_test = flatten_all_images(images_test)

        dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=self.hidden, output_neurons=10)

        dnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=images_test,
            labels_test=self.labels_test, images_train=images_train, labels_train=self.labels_train,
            model_name=model_name
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")

    def model_7(self):
        model_name = "cnn_fully_connected_no_hidden"
        print("\n\n")
        print(f"=======---------model (({model_name})) has started---------=======")
        print("\n\n")

        images_train = reshape_all_images(self.images_train)
        images_test = reshape_all_images(self.images_test)

        images_train = self.cnn.convolve(images_train)
        # adding pooling layer seems to improve the accuracy
        images_train = self.cnn.pool(images_train)
        images_train = flatten_all_images(images_train)

        images_test = self.cnn.convolve(images_test)
        images_test = self.cnn.pool(images_test)
        images_test = flatten_all_images(images_test)

        fnn = fnn_layers(input_neurons=images_train.shape[1], output_neurons=10)
        fnn.evaluate_model(
            epoch_size=self.epoch, alpha=self.alpha, images_test=images_test,
            labels_test=self.labels_test, images_train=images_train, labels_train=self.labels_train,
            model_name="fnn"
        )
        print("\n\n")
        print(f"=======---------model (({model_name})) finished---------=======")
        print("\n\n")


