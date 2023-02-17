import pandas as pd
import matplotlib.pyplot as plt


def read_mnist_data(dataset_path, rows_count):

    data = pd.read_csv(dataset_path, header=None, nrows=rows_count).values
    images, labels = data[:, 1:], data[:, 0]

    # Normalize the image features
    images = images / 255.0

    return images, labels
