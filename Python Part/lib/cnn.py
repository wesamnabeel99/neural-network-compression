import numpy as np


class cnn_layers():
    def __int__(self, x):
        self.x = x

    def convolve(self, image, kernel):
        # Get the dimensions of the input image and kernel
        image_height, image_width = image.shape
        kernel_height, kernel_width = kernel.shape

        # memory for the image
        output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype="float32")
        # Loop over the output image, sliding the kernel across
        for i in range(image_height - kernel_height + 1):
            for j in range(image_width - kernel_width + 1):
                output[i, j] = (image[i:i + kernel_height, j:j + kernel_width] * kernel).sum()
        return output

    def pool(self, convolved_image):
        stride = 2
        pool_output = []
        for i in range(0, convolved_image.shape[0], stride):
            row = []
            for j in range(0, convolved_image.shape[1], stride):
                row.append(np.amax(convolved_image[i:i + stride, j:j + stride]))
            pool_output.append(row)
        return np.array(pool_output)





