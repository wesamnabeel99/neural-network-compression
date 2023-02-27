import numpy as np


def reshape_all_images(one_d_images):
    image_size = 28
    return one_d_images.reshape(one_d_images.shape[0], image_size, image_size)


def flatten_all_images(two_d_images):
    return two_d_images.reshape(two_d_images.shape[0], -1)
