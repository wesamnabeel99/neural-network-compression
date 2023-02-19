from lib.cnn import cnn_layers
from lib.dnn_layers import dnn_layers
from utils.image_helper import *


def model_one(images_train, labels_train, images_test, labels_test):
    model_name = "cnn_pooling_dnn"

    print(f"=======---------model {model_name} has started---------=======")
    images_train = reshape_all_images(images_train)
    images_test = reshape_all_images(images_test)

    cnn = cnn_layers(n_kernels=3)

    images_train = cnn.convolve(images_train)
    images_train = cnn.pool(images_train)
    images_train = flatten_all_images(images_train)

    images_test = cnn.convolve(images_test)
    images_test = cnn.pool(images_test)
    images_test = flatten_all_images(images_test)

    dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=100, output_neurons=10)

    dnn.evaluate_model(
        epoch_size=50, alpha=0.1, images_test=images_test,
        labels_test=labels_test, images_train=images_train, labels_train=labels_train, model_name=model_name
    )
    print(f"=======---------model {model_name} finished---------=======")


def model_two(images_train, labels_train, images_test, labels_test):
    model_name = "cnn_dnn"
    print(f"=======---------model {model_name} has started---------=======")

    images_train = reshape_all_images(images_train)
    images_test = reshape_all_images(images_test)

    cnn = cnn_layers(n_kernels=3)

    images_train = cnn.convolve(images_train)
    images_train = flatten_all_images(images_train)

    images_test = cnn.convolve(images_test)
    images_test = flatten_all_images(images_test)

    dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=100, output_neurons=10)

    dnn.evaluate_model(
        epoch_size=50, alpha=0.1, images_test=images_test,
        labels_test=labels_test, images_train=images_train, labels_train=labels_train, model_name=model_name
    )
    print(f"=======---------model {model_name} finished---------=======")


def model_three(images_train, labels_train, images_test, labels_test):
    model_name = "dnn"
    print(f"=======---------model {model_name} has started---------=======")

    dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=100, output_neurons=10)

    dnn.evaluate_model(
        epoch_size=50, alpha=0.1, images_test=images_test,
        labels_test=labels_test, images_train=images_train, labels_train=labels_train, model_name="dnn"
    )
    print(f"=======---------model {model_name} finished---------=======")


def model_four(images_train, labels_train, images_test, labels_test):
    model_name = "pooling_dnn"
    print(f"=======---------model {model_name} has started---------=======")

    images_train = reshape_all_images(images_train)
    images_test = reshape_all_images(images_test)

    cnn = cnn_layers(n_kernels=3)

    images_train = cnn.pool(images_train)
    images_train = flatten_all_images(images_train)

    images_test = cnn.pool(images_test)
    images_test = flatten_all_images(images_test)

    dnn = dnn_layers(input_neurons=images_train.shape[1], hidden_neurons=100, output_neurons=10)

    dnn.evaluate_model(
        epoch_size=50, alpha=0.1, images_test=images_test,
        labels_test=labels_test, images_train=images_train, labels_train=labels_train, model_name=model_name
    )
    print(f"=======---------model {model_name} finished---------=======")


def model_five(images_test, labels_test):
    model_name = "cnn_dnn_forward"
    print(f"=======---------model {model_name} has started---------=======")

    images_test = reshape_all_images(images_test)

    cnn = cnn_layers(n_kernels=3)

    images_test = cnn.convolve(images_test)
    images_test = flatten_all_images(images_test)

    dnn = dnn_layers(input_neurons=images_test.shape[1], hidden_neurons=100, output_neurons=10)

    accuracy = dnn.forward(images_test, labels_test)
    print(f"model {model_name} accuracy: {accuracy}")
    print(f"=======---------model {model_name} finished---------=======")
