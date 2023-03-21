import pandas as pd

def read_mnist_data(dataset_path, rows_count):

    data = pd.read_csv(dataset_path, header=0, nrows=rows_count).values
    images, labels = data[:, 1:], data[:, 0]

    # Normalize the image features
    images = images / 255.0

    return images, labels
