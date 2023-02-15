import pandas as pd
import matplotlib.pyplot as plt


def read_mnist_data(dataset_path, rows_count):

    data = pd.read_csv(dataset_path, header=None, nrows=rows_count).values
    images, labels = data[:, 1:], data[:, 0]

    # Normalize the image features
    images = images / 255.0
    # Taking sample image from the dataset
    sample = images[[0], :].reshape(28, 28)
    plt.imshow(sample)
    plt.savefig("temp/sample_image.png")

    return images, labels
