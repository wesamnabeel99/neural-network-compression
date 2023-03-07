import numpy as np


class cnn_layers():
    def __init__(self, n_kernels, kernel_size):
        # Setting the shapes and sizes for convolution operation
        self.kernel = np.random.uniform(0, 1, size=(kernel_size, kernel_size))
        self.n_kernels = n_kernels
        # normalize the kernel
        kernel_sum = np.sum(self.kernel)
        if kernel_sum != 0:
            self.kernel = self.kernel / kernel_sum

    def convolve(self, images):
        print("---convolution start---")
        convolution_images = []
        for image in images:
            image_height, image_width = image.shape
            kernel_height, kernel_width = self.kernel.shape[0], self.kernel.shape[0]
            output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype="float32")
            for i in range(image_height - kernel_height + 1):
                for j in range(image_width - kernel_width + 1):
                    output[i, j] = (image[i:i + kernel_height, j:j + kernel_width] * self.kernel).sum()
            convolution_images.append(output)
        print("###convolution end###")
        return np.array(convolution_images)

    def convolve_multiple_kernels(self, images):
        print("---convolution start---")
        convolved_images = []
        for kernel in range(self.n_kernels):

            convolution_images = []
            for image in images:
                image_height, image_width = image.shape
                kernel_height, kernel_width = self.kernel.shape[0], self.kernel.shape[0]
                output = np.zeros((image_height - kernel_height + 1, image_width - kernel_width + 1), dtype="float32")
                for i in range(image_height - kernel_height + 1):
                    for j in range(image_width - kernel_width + 1):
                        output[i, j] = (image[i:i + kernel_height, j:j + kernel_width] * self.kernel).sum()
                convolution_images.append(output)
            print("###convolution end###")
            convolved_images.append(np.array(convolution_images))
            self.update_kernel()

        return np.concatenate(convolved_images, axis=1)

    def pool(self, convolved_images):
        print("---pooling start---")
        images = []
        for image in convolved_images:
            stride = 2
            pool_output = []
            for i in range(0, image.shape[0], stride):
                row = []
                for j in range(0, image.shape[1], stride):
                    row.append(np.amax(image[i:i + stride, j:j + stride]))
                pool_output.append(row)
            images.append(np.array(pool_output))
        print("###pooling end###")
        return np.array(images)

    def update_kernel(self):
        # update the kernel
        self.kernel = np.random.uniform(0, 1, size=(self.n_kernels, self.n_kernels))
        # normalize the kernel
        kernel_sum = np.sum(self.kernel)
        if kernel_sum != 0:
            self.kernel = self.kernel / kernel_sum
